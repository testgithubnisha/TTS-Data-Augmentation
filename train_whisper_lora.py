import argparse
import json
import math
import random
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import whisper
from whisper.tokenizer import get_tokenizer

PROJECT = Path(r"C:\Users\HP\Documents\AI project")
DATA_DIR = PROJECT / "asr_prepared" / "run_20261007_231727_449861"
CHECKPOINT = PROJECT / "whisper_models" / "small.pt"
OUTPUT_ROOT = PROJECT / "whisper_runs"
SAMPLE_RATE = 16000



class LoRALinear(nn.Module):
    """Frozen original linear layer plus trainable low-rank adapter."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.base = base

        # Freeze the original pretrained layer.
        for p in self.base.parameters():
            p.requires_grad = False

        # Create LoRA layers directly on the base layer's device and dtype.
        self.lora_A = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )

        self.lora_B = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )

        # Initialize A randomly and B to zero.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lora_output = self.lora_B(
            self.dropout(self.lora_A(x))
        )
        return self.base(x) + lora_output * self.scale


def add_lora(model, rank=8, alpha=16, dropout=0.05):
    replaced = 0

    for block_group in (model.encoder.blocks, model.decoder.blocks):
        for block in block_group:
            attentions = [block.attn]

            if getattr(block, "cross_attn", None) is not None:
                attentions.append(block.cross_attn)

            for attn in attentions:
                for attr in ("query", "key", "value", "out"):
                    layer = getattr(attn, attr, None)

                    if isinstance(layer, nn.Linear):
                        setattr(
                            attn,
                            attr,
                            LoRALinear(
                                layer,
                                rank=rank,
                                alpha=alpha,
                                dropout=dropout,
                            ),
                        )
                        replaced += 1

    if replaced == 0:
        raise RuntimeError("No attention projection layers were adapted.")

    return replaced



class JsonlWhisperDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, n_mels: int, text_context: int):
        self.jsonl_path = jsonl_path
        self.rows = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    row = json.loads(line)
                    if not row.get("audio_path") or not isinstance(row.get("text"), str):
                        raise ValueError(f"Invalid JSONL row at line {line_no}: expected audio_path and text")
                    self.rows.append(row)
        self.tokenizer = tokenizer
        self.n_mels = n_mels
        self.text_context = text_context
        if not self.rows:
            raise ValueError(f"No records found in {jsonl_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = DATA_DIR / audio_path
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if sr != SAMPLE_RATE:
            # Dataset is expected to be 16 kHz; fail clearly rather than silently resample incorrectly.
            raise ValueError(f"Expected 16 kHz audio but got {sr} Hz: {audio_path}")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)

        text_tokens = self.tokenizer.encode(row["text"].strip())
        prefix = list(self.tokenizer.sot_sequence)
        max_text_tokens = self.text_context - len(prefix) - 1  # reserve EOT
        text_tokens = text_tokens[:max_text_tokens]
        tokens = prefix + text_tokens + [self.tokenizer.eot]
        return mel, torch.tensor(tokens, dtype=torch.long)


def collate_batch(items, pad_token):
    mels = torch.stack([item[0] for item in items])
    sequences = [item[1] for item in items]
    max_len = max(len(s) for s in sequences)
    decoder_in = torch.full((len(items), max_len - 1), pad_token, dtype=torch.long)
    labels = torch.full((len(items), max_len - 1), -100, dtype=torch.long)
    for i, seq in enumerate(sequences):
        n = len(seq) - 1
        decoder_in[i, :n] = seq[:-1]
        labels[i, :n] = seq[1:]
    return mels, decoder_in, labels



def enable_encoder_gradient_checkpointing(model):
    """Recompute encoder blocks during backward to reduce VRAM use."""
    encoder = model.encoder

    def forward_with_checkpointing(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + self.positional_embedding).to(x.dtype)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.ln_post(x)

    encoder.forward = types.MethodType(forward_with_checkpointing, encoder)


def save_adapter(model, optimizer, out_dir: Path, step: int, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
    torch.save({
        "adapter_state_dict": adapter,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "base_checkpoint": str(CHECKPOINT),
        "dataset": args.dataset,
    }, out_dir / "adapter_checkpoint.pt")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune local OpenAI Whisper small.pt using LoRA adapters.")
    parser.add_argument("--dataset", choices=["train_real.jsonl", "train_augmented.jsonl"], required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--save_every", type=int, default=250, help="Save after this many optimizer steps")
    parser.add_argument("--max_steps", type=int, default=0, help="Optional short run; 0 means full dataset")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This script is configured for the RTX 4050 GPU.")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Local Whisper checkpoint not found: {CHECKPOINT}")
    jsonl_path = DATA_DIR / args.dataset
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    out_dir = OUTPUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LOCAL OpenAI Whisper checkpoint: {CHECKPOINT}", flush=True)
    model = whisper.load_model(str(CHECKPOINT), device=device)
    model.train()
    # Freeze every pretrained parameter first; only newly inserted LoRA matrices will train.
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = get_tokenizer(multilingual=True, language="zh", task="transcribe")
    replaced = add_lora(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    enable_encoder_gradient_checkpointing(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Attention projections adapted: {replaced}", flush=True)
    print("Encoder gradient checkpointing: enabled", flush=True)
    print(f"Trainable parameters: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.3f}%)", flush=True)

    dataset = JsonlWhisperDataset(jsonl_path, tokenizer, model.dims.n_mels, model.dims.n_text_ctx)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
        collate_fn=lambda items: collate_batch(items, tokenizer.eot),
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    batches_seen = 0
    running_loss = 0.0
    max_batches = args.max_steps * args.grad_accum if args.max_steps > 0 else 0
    print(f"Dataset: {jsonl_path}", flush=True)
    print(f"Training records: {len(dataset)} | epochs: {args.epochs} | batch: {args.batch_size} | grad accumulation: {args.grad_accum}", flush=True)
    print("Starting training...", flush=True)

    for epoch in range(args.epochs):
        for batch_idx, (mels, decoder_in, labels) in enumerate(loader):
            if max_batches and batches_seen >= max_batches:
                break
            mels = mels.to(device, non_blocking=True)
            decoder_in = decoder_in.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                audio_features = model.encoder(mels)
                logits = model.decoder(decoder_in, audio_features)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            batches_seen += 1
            running_loss += float(loss.detach().cpu())
            last_batch = (batch_idx + 1 == len(loader)) or (max_batches and batches_seen >= max_batches)
            if batches_seen % args.grad_accum == 0 or last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    avg_loss = running_loss / max(1, min(10 * args.grad_accum, batches_seen))
                    print(f"epoch={epoch+1} optimizer_step={global_step} batches={batches_seen} recent_loss_avg={avg_loss:.4f} gpu_allocated_GB={torch.cuda.memory_allocated()/1024**3:.2f}", flush=True)
                    running_loss = 0.0
                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_adapter(model, optimizer, out_dir, global_step, args)
                    print(f"Saved checkpoint at optimizer step {global_step}", flush=True)
        if max_batches and batches_seen >= max_batches:
            break

    save_adapter(model, optimizer, out_dir, global_step, args)
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "run_name": args.run_name,
            "dataset": args.dataset,
            "records": len(dataset),
            "epochs_requested": args.epochs,
            "batches_seen": batches_seen,
            "optimizer_steps": global_step,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "rank": args.rank,
            "alpha": args.alpha,
            "base_checkpoint": str(CHECKPOINT),
            "method": "LoRA fine-tuning of OpenAI Whisper small attention projections",
        }, f, indent=2)
    print(f"TRAINING COMPLETE. Adapter and summary saved in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
