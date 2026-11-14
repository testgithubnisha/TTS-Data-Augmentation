import argparse
import csv
import gc
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import whisper

try:
    import jieba
except ImportError:
    jieba = None

PROJECT = Path(r"C:\Users\HP\Documents\AI project")
DATA_DIR = PROJECT / "asr_prepared" / "run_20261007_231727_449861"
CHECKPOINT = PROJECT / "whisper_models" / "small.pt"
OUTPUT_DIR = PROJECT / "whisper_runs" / "model_comparison"
SAMPLE_RATE = 16000


def add_lora(model, rank=8, alpha=16, dropout=0.05):
    """Re-create the same LoRA adapter layout used by train_whisper_lora.py."""
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
                            LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout),
                        )
                        replaced += 1
    if replaced == 0:
        raise RuntimeError("No attention projection layers were adapted.")
    return replaced


class LoRALinear(nn.Module):
    """Frozen original linear layer plus trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
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
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.dropout(self.lora_A(x))) * self.scale


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("audio_path") or not isinstance(row.get("text"), str):
                raise ValueError(
                    f"Invalid JSONL row at line {line_number}: expected audio_path and text"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No test records found in {path}")
    return rows


def resolve_audio_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = DATA_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def normalize_for_cer(text: str) -> str:
    """Keep Unicode letters/numbers; ignore punctuation and whitespace for CER."""
    return "".join(
        ch.lower()
        for ch in text
        if unicodedata.category(ch)[:1] in {"L", "N"}
    )


def normalize_for_words(text: str) -> str:
    """Remove punctuation while retaining letters, numbers and word spacing."""
    out = []
    for ch in text.lower():
        category = unicodedata.category(ch)[:1]
        if category in {"L", "N"}:
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def edit_distance(reference, hypothesis) -> int:
    """Levenshtein distance using O(len(hypothesis)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def get_word_tokens(text: str, language: str):
    cleaned = normalize_for_words(text)
    if language.lower().startswith("zh"):
        # WER for Chinese needs word segmentation; jieba provides consistent segmentation.
        return [token for token in jieba.lcut(cleaned, cut_all=False) if token.strip()]
    return cleaned.split()


def load_model(run_name, device, include_adapter=True):
    print(f"Loading original checkpoint for {run_name or 'base Whisper'}...", flush=True)
    model = whisper.load_model(str(CHECKPOINT), device=device)

    if include_adapter:
        adapter_path = PROJECT / "whisper_runs" / run_name / "adapter_checkpoint.pt"
        if not adapter_path.is_file():
            raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")

        # Every experiment is loaded into a fresh copy of the same original Whisper checkpoint.
        payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
        adapter_state = payload.get("adapter_state_dict")
        if not isinstance(adapter_state, dict) or not adapter_state:
            raise ValueError(f"No adapter_state_dict found in {adapter_path}")
        rank = int(payload.get("rank", 8))
        alpha = int(payload.get("alpha", 16))
        dropout = float(payload.get("dropout", 0.05))
        replaced = add_lora(model, rank=rank, alpha=alpha, dropout=dropout)
        model_state_keys = set(model.state_dict().keys())
        unexpected = [key for key in adapter_state if key not in model_state_keys]
        if unexpected:
            raise RuntimeError(
                "Adapter keys do not match this model layout. First unexpected keys: "
                + ", ".join(unexpected[:5])
            )
        incompatible = model.load_state_dict(adapter_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected adapter weights: " + ", ".join(incompatible.unexpected_keys[:5])
            )
        matched = sum(1 for key in adapter_state if "lora_A" in key or "lora_B" in key)
        if matched != len(adapter_state):
            raise RuntimeError("The checkpoint contains unexpected non-LoRA weights.")
        print(
            f"Loaded {matched} adapter tensors; attention projections adapted: {replaced}",
            flush=True,
        )

    # Make the base Whisper model and all inserted LoRA adapters use one dtype.
    # Whisper decoding sends FP16 features when fp16=True on CUDA; casting the
    # complete model prevents Half/Float mismatches between base layers and adapters.
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device=device, dtype=torch.float32)
    print(f"Evaluation model dtype: {next(model.parameters()).dtype}", flush=True)
    model.eval()
    return model


def evaluate_condition(run_name, label, rows, language, device, detailed_csv_path):
    model = load_model(run_name, device, include_adapter=(run_name is not None))
    total_char_errors = 0
    total_reference_chars = 0
    total_word_errors = 0
    total_reference_words = 0
    started = time.perf_counter()

    with detailed_csv_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "sample_index",
                "audio_path",
                "reference",
                "prediction",
                "reference_characters",
                "character_errors",
                "CER",
                "reference_words",
                "word_errors",
                "WER",
            ],
        )
        for index, row in enumerate(rows, start=1):
            audio_path = resolve_audio_path(row["audio_path"])
            audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sample_rate != SAMPLE_RATE:
                raise ValueError(
                    f"Expected 16 kHz audio but got {sample_rate} Hz: {audio_path}"
                )

            with torch.inference_mode():
                result = model.transcribe(
                    np.asarray(audio, dtype=np.float32),
                    language=language,
                    task="transcribe",
                    fp16=(device.type == "cuda"),
                    verbose=False,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
            prediction = result.get("text", "").strip()
            reference = row["text"].strip()

            ref_chars = normalize_for_cer(reference)
            hyp_chars = normalize_for_cer(prediction)
            char_errors = edit_distance(list(ref_chars), list(hyp_chars))
            char_denominator = len(ref_chars)
            sample_cer = (char_errors / char_denominator) if char_denominator else (0.0 if not hyp_chars else 1.0)

            ref_words = get_word_tokens(reference, language)
            hyp_words = get_word_tokens(prediction, language)
            word_errors = edit_distance(ref_words, hyp_words)
            word_denominator = len(ref_words)
            sample_wer = (word_errors / word_denominator) if word_denominator else (0.0 if not hyp_words else 1.0)

            total_char_errors += char_errors
            total_reference_chars += char_denominator
            total_word_errors += word_errors
            total_reference_words += word_denominator

            writer.writerow(
                {
                    "model": label,
                    "sample_index": index,
                    "audio_path": str(audio_path),
                    "reference": reference,
                    "prediction": prediction,
                    "reference_characters": char_denominator,
                    "character_errors": char_errors,
                    "CER": f"{sample_cer:.8f}",
                    "reference_words": word_denominator,
                    "word_errors": word_errors,
                    "WER": f"{sample_wer:.8f}",
                }
            )
            csv_file.flush()

            if index == 1 or index % 10 == 0 or index == len(rows):
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0 else 0.0
                remaining = (len(rows) - index) / rate if rate > 0 else float("nan")
                print(
                    f"[{label}] {index}/{len(rows)} samples | "
                    f"elapsed {elapsed/60:.1f} min | approx. remaining {remaining/60:.1f} min",
                    flush=True,
                )

    summary = {
        "model": label,
        "run_name": run_name or "original_whisper_small",
        "samples": len(rows),
        "WER": total_word_errors / total_reference_words if total_reference_words else 0.0,
        "CER": total_char_errors / total_reference_chars if total_reference_chars else 0.0,
        "word_errors": total_word_errors,
        "reference_words": total_reference_words,
        "character_errors": total_char_errors,
        "reference_characters": total_reference_chars,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "language": language,
        "test_file": str(DATA_DIR / "test_real.jsonl"),
        "scoring_normalization": "lowercase; punctuation and whitespace excluded from CER; jieba segmentation for Chinese WER",
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Experiment A and B adapters on the same held-out real test set."
    )
    parser.add_argument("--test_file", default="test_real.jsonl")
    parser.add_argument("--language", default="zh", help="Whisper language code; default: zh")
    parser.add_argument(
        "--include_baseline",
        action="store_true",
        help="Also evaluate the untouched original Whisper small checkpoint (takes extra time).",
    )
    args = parser.parse_args()

    if jieba is None and args.language.lower().startswith("zh"):
        print(
            "Missing dependency: jieba (needed for Chinese word-level WER).\n"
            "Install it in this project environment with:\n"
            '  & ".\\.venv_cosyvoice\\Scripts\\python.exe" -m pip install jieba',
            file=sys.stderr,
        )
        raise SystemExit(2)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Original Whisper checkpoint not found: {CHECKPOINT}")
    test_path = DATA_DIR / args.test_file
    if not test_path.is_file():
        raise FileNotFoundError(f"Test JSONL file not found: {test_path}")

    rows = read_jsonl(test_path)
    if not torch.cuda.is_available():
        print("CUDA not available; evaluation will run on CPU and may be very slow.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = OUTPUT_DIR / "predictions.csv"
    summary_csv_path = OUTPUT_DIR / "comparison_summary.csv"
    summary_json_path = OUTPUT_DIR / "comparison_summary.json"

    with predictions_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model", "sample_index", "audio_path", "reference", "prediction",
                "reference_characters", "character_errors", "CER",
                "reference_words", "word_errors", "WER",
            ]
        )

    conditions = []
    if args.include_baseline:
        conditions.append((None, "original_whisper_small"))
    conditions.extend(
        [
            ("experiment_A_real", "experiment_A_real"),
            ("experiment_B_augmented", "experiment_B_augmented"),
        ]
    )

    print(f"Test set: {test_path}", flush=True)
    print(f"Test records: {len(rows)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Detailed predictions will be saved to: {predictions_path}", flush=True)

    summaries = []
    for run_name, label in conditions:
        summary = evaluate_condition(run_name, label, rows, args.language, device, predictions_path)
        summaries.append(summary)
        print(
            f"Completed {label}: WER={summary['WER']:.4%}, CER={summary['CER']:.4%}",
            flush=True,
        )

    by_name = {summary["model"]: summary for summary in summaries}
    if "experiment_A_real" in by_name and "experiment_B_augmented" in by_name:
        a_cer = by_name["experiment_A_real"]["CER"]
        b_cer = by_name["experiment_B_augmented"]["CER"]
        a_wer = by_name["experiment_A_real"]["WER"]
        b_wer = by_name["experiment_B_augmented"]["WER"]
        comparison = {
            "CER_relative_improvement_percent_B_vs_A": ((a_cer - b_cer) / a_cer * 100) if a_cer else None,
            "WER_relative_improvement_percent_B_vs_A": ((a_wer - b_wer) / a_wer * 100) if a_wer else None,
            "interpretation": "Positive relative improvement means Experiment B has lower error; negative means Experiment B has higher error.",
        }
    else:
        comparison = {}

    output = {
        "test_file": str(test_path),
        "test_records": len(rows),
        "language": args.language,
        "models": summaries,
        "comparison_B_vs_A": comparison,
        "notes": [
            "All models were evaluated on the same held-out real test records.",
            "WER for Chinese uses jieba word segmentation; CER is character-level.",
            "Punctuation and whitespace are excluded during scoring normalization.",
            "WER/CER values are error rates; lower is better.",
        ],
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    with summary_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "model", "run_name", "samples", "WER", "CER", "word_errors",
            "reference_words", "character_errors", "reference_characters", "elapsed_minutes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})

    print("\nCOMPARISON COMPLETE", flush=True)
    for summary in summaries:
        print(
            f"{summary['model']}: WER={summary['WER']:.4%} | CER={summary['CER']:.4%}",
            flush=True,
        )
    if comparison:
        print(
            "Experiment B relative improvement vs A: "
            f"WER={comparison['WER_relative_improvement_percent_B_vs_A']:.2f}% | "
            f"CER={comparison['CER_relative_improvement_percent_B_vs_A']:.2f}%",
            flush=True,
        )
    print(f"Summary CSV: {summary_csv_path}", flush=True)
    print(f"Summary JSON: {summary_json_path}", flush=True)
    print(f"Per-sample predictions: {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
