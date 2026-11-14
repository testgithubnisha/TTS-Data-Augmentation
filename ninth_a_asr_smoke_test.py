"""Step 9A: one disposable decoder-only LoRA update on cached Whisper small.
No checkpoint is saved. Existing ASR/TTS weights and data remain unchanged.
"""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'asr_prepared' / 'run_20261007_231727_449861')
    args = parser.parse_args()
    data = args.data_dir.resolve()
    summary = json.loads((data / 'preparation_summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'complete' or summary['sample_rate'] != 16000:
        raise RuntimeError('Step 8B must be complete with 16 kHz audio.')

    print('STEP 9A: WHISPER ASR TRAINING SMOKE TEST', flush=True)
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import whisper
    from whisper.tokenizer import get_tokenizer
    from peft import LoraConfig, inject_adapter_in_model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run with .venv_cosyvoice Python.')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    print('PyTorch:', torch.__version__, flush=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    cached = ROOT / 'whisper_models' / 'small.pt'
    if not cached.is_file():
        raise FileNotFoundError(f'Expected cached Whisper weights: {cached}. No download attempted.')
    print('Loading cached Whisper small on CPU...', flush=True)
    model = whisper.load_model(str(cached), device='cpu')
    tokenizer = get_tokenizer(multilingual=True, language='zh', task='transcribe')
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    manifest = data / 'train_augmented.jsonl'
    rows = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(rows) != summary['counts']['train_augmented']:
        raise RuntimeError('Training manifest count differs from Step 8B summary.')
    # Test the longest tokenized training target, not just a short easy sample.
    # Validation/test data are not opened or scored here.
    chosen = None
    longest = -1
    for row in rows:
        if row['split'] != 'train' or not row['text'].strip():
            raise RuntimeError(f"Invalid training record: {row['id']}")
        seq = prefix + tokenizer.encode(row['text']) + [tokenizer.eot]
        length = len(seq) - 1
        if length > model.dims.n_text_ctx:
            raise RuntimeError(f"{row['id']} needs {length} decoder positions; maximum {model.dims.n_text_ctx}. No labels were truncated.")
        if row['duration_seconds'] > 30:
            raise RuntimeError(f"{row['id']} exceeds 30 seconds. Explicit segmentation is required.")
        if length > longest:
            chosen, longest = (row, seq), length
    if chosen is None:
        raise RuntimeError('Empty training manifest.')
    row, seq = chosen
    path = (data / row['audio_path']).resolve()
    if not path.is_relative_to(data):
        raise RuntimeError('Audio path escapes the prepared folder.')
    waveform, sr = sf.read(str(path), dtype='float32')
    if sr != 16000 or waveform.ndim != 1 or not 0 < len(waveform) <= 480000:
        raise RuntimeError('Expected nonempty mono 16 kHz audio of at most 30 seconds.')
    if not np.isfinite(waveform).all():
        raise RuntimeError('Non-finite audio.')
    print('Sample:', row['id'], flush=True)
    print('Text:', row['text'], flush=True)
    print('Decoder input tokens:', longest, flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    settings = dict(r=8, lora_alpha=16, target_modules=['query', 'value'],
                    lora_dropout=0.05, bias='none')
    # Inject into decoder only: includes decoder self- and cross-attention.
    model.decoder = inject_adapter_in_model(LoraConfig(**settings), model.decoder)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith('decoder.') or 'lora_' not in n for n, p in trainable):
        raise RuntimeError('Unexpected trainable parameters: only decoder LoRA is allowed.')
    print('Trainable parameters:', sum(p.numel() for _, p in trainable), flush=True)
    print('Encoder frozen; batch size 1; no checkpoint will be saved.', flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print('Autocast precision:', dtype, flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = model.to('cuda')
    model.encoder.eval()
    model.decoder.train()
    # Keep trainable master parameters float32 and use mixed precision compute.
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    audio = whisper.pad_or_trim(torch.from_numpy(waveform)).to('cuda')
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).unsqueeze(0)
    input_ids = torch.tensor([seq[:-1]], dtype=torch.long, device='cuda')
    labels = torch.tensor([seq[1:]], dtype=torch.long, device='cuda')
    # Only score transcript tokens + EOT, not the forced language/task prefix.
    labels[:, :len(prefix) - 1] = -100
    optimizer.zero_grad(set_to_none=True)
    print('Running encoder and decoder forward pass...', flush=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        features = model.encoder(mel)
    with torch.autocast('cuda', dtype=dtype):
        logits = model.decoder(input_ids, features)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss.')
    print('Loss:', float(loss.detach()), flush=True)
    print('Running backward and optimizer step...', flush=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
    if float(norm) == 0:
        raise RuntimeError('All adapter gradients are zero.')
    selected = next((n, p) for n, p in trainable if p.grad is not None and bool(torch.any(p.grad != 0)))
    name, parameter = selected
    before = parameter.detach().clone()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    change = float((parameter.detach() - before).abs().max())
    if change == 0:
        raise RuntimeError('Optimizer did not update the checked adapter tensor.')
    report = {
        'status': 'passed', 'model': 'whisper-small', 'model_path': str(cached),
        'data_dir': str(data), 'train_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'sample_id': row['id'], 'decoder_input_tokens': longest,
        'architecture': 'Frozen encoder/base decoder; decoder self- and cross-attention query/value LoRA',
        'lora_settings': settings, 'batch_size': 1, 'seed': 42,
        'trainable_parameters': sum(p.numel() for _, p in trainable),
        'precision': str(dtype), 'loss': float(loss.detach()), 'gradient_norm': float(norm),
        'checked_adapter': name, 'max_parameter_change': change,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'checkpoint_saved': False, 'validation_or_test_evaluated': False,
        'limitation': 'One-batch engineering check, not model quality or an improvement result.'
    }
    out = ROOT / 'asr_smoke_tests'
    out.mkdir(exist_ok=True)
    dest = out / datetime.now().strftime('smoke_%Y%m%d_%H%M%S_%f.json')
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSTEP 9A PASSED', flush=True)
    print(json.dumps(report, indent=2))
    print('Report:', dest)
    print('Disposable update completed. No checkpoint saved. Full training has not started.')

if __name__ == '__main__':
    main()
