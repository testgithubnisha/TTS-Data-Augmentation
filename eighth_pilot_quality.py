"""Step 8A: audit pilot audio and flag ASR mismatches for review.
Does not delete audio, change transcripts, or automatically approve training data.
Requires sixth_auto_check.py beside this file for the same edit-distance logic.
"""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = run / "manifest.jsonl"
    print("STEP 8A: PILOT QUALITY CHECK", flush=True)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Manifest is empty.")
    ids = [r["synthetic_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate synthetic IDs in manifest.")
    paths = []
    for r in records:
        if r.get("source_split") != "train" or r.get("is_synthetic") is not True:
            raise ValueError(f"Unexpected source/synthetic flag: {r['synthetic_id']}")
        p = (run / r["audio_path"]).resolve()
        if not p.is_relative_to(run):
            raise ValueError("Audio path must be inside the pilot folder.")
        paths.append(p)
    if len(set(paths)) != len(paths):
        raise ValueError("Multiple manifest rows point to the same audio.")

    print("Loading audio and ASR packages...", flush=True)
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    from scipy.signal import resample_poly
    from opencc import OpenCC
    from sixth_auto_check import units, align

    cc = OpenCC("t2s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Files:", len(records), "Device:", device, flush=True)
    cache = ROOT / "whisper_models"
    print("Loading Whisper small from:", cache, flush=True)
    model = whisper.load_model("small", device=device, download_root=str(cache))
    model.eval()
    out = run / datetime.now().strftime("quality_%Y%m%d_%H%M%S_%f")
    out.mkdir()
    fields = ["synthetic_id", "source_id", "text", "audio_path", "duration_seconds",
              "sample_rate", "peak", "rms", "full_scale_fraction", "technical_status",
              "technical_notes", "asr_text", "normalized_target", "normalized_asr",
              "reference_units", "substitutions", "deletions", "insertions",
              "mixed_error_rate_percent", "review_priority", "asr_alignment_json",
              "human_decision", "human_notes"]
    summary = {"status": "running", "manifest_rows": len(records), "processed": 0,
               "technical_failures": 0, "technical_review_flags": 0,
               "asr_scored": 0, "asr_exact_matches": 0, "asr_mismatches": 0,
               "reference_units": 0, "errors": 0, "total_audio_seconds": 0,
               "automatically_approved": 0,
               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "asr_model": "small", "language": "zh", "task": "transcribe",
               "normalization": "OpenCC t2s, then Step 6 NFKC/casefold Han-character + English-word units; no translation equivalence",
               "technical_flag_rules": "RMS <0.001, any full-scale PCM16 samples, duration >30 seconds, or manifest metadata mismatch: review; not automatic rejection",
               "limitation": "ASR flags are not confirmed speech errors. Human decisions remain blank. No pronunciation or naturalness ratings inferred."}
    def save_summary():
        (out / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_summary()
    try:
        with (out / "quality_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            handle.flush()
            for index, (record, path) in enumerate(zip(records, paths), 1):
                print(f"[{index}/{len(records)}] {record['synthetic_id']}", flush=True)
                row = {"synthetic_id": record["synthetic_id"], "source_id": record["source_id"],
                       "text": record["text"], "audio_path": str(path)}
                flags = []
                try:
                    samples, sr = sf.read(path, dtype="float32", always_2d=True)
                    if not samples.size or not np.isfinite(samples).all():
                        raise ValueError("Empty or non-finite audio")
                    peak = float(np.abs(samples).max())
                    if peak == 0:
                        raise ValueError("Entirely silent audio")
                    duration = len(samples) / sr
                    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    full_scale = float(np.mean(np.abs(samples) >= 32767 / 32768))
                    if rms < 0.001:
                        flags.append("very_low_level_review")
                    if full_scale > 0:
                        flags.append("full_scale_samples_review")
                    if duration > 30:
                        flags.append("longer_than_30_seconds_review")
                    if sr != record["sample_rate"] or abs(duration - record["duration_seconds"]) > 0.02:
                        flags.append("manifest_audio_metadata_mismatch")
                    target_units = units(cc.convert(record["text"]))
                    if not target_units:
                        raise ValueError("Empty normalized target")
                    row.update(duration_seconds=round(duration, 4), sample_rate=sr,
                               peak=round(peak, 6), rms=round(rms, 6),
                               full_scale_fraction=full_scale, technical_status="review" if flags else "pass",
                               technical_notes="; ".join(flags))
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    row.update(technical_status="fail", technical_notes=str(exc), review_priority="technical_failure")
                    summary["technical_failures"] += 1
                    writer.writerow(row)
                    handle.flush()
                    summary["processed"] += 1
                    save_summary()
                    continue
                summary["total_audio_seconds"] += duration
                summary["technical_review_flags"] += bool(flags)
                audio = samples.mean(axis=1)
                if sr != 16000:
                    divisor = math.gcd(sr, 16000)
                    audio = resample_poly(audio, 16000 // divisor, sr // divisor)
                audio = np.ascontiguousarray(audio, dtype=np.float32)
                with torch.inference_mode():
                    result = model.transcribe(audio, language="zh", task="transcribe", temperature=0.0,
                        beam_size=1, condition_on_previous_text=False, fp16=(device == "cuda"), verbose=None)
                hypothesis = result["text"].strip()
                hypothesis_units = units(cc.convert(hypothesis))
                counts, edits = align(target_units, hypothesis_units)
                errors = sum(counts.values())
                row.update(asr_text=hypothesis, normalized_target=" ".join(target_units),
                           normalized_asr=" ".join(hypothesis_units), reference_units=len(target_units),
                           **counts, mixed_error_rate_percent=round(100 * errors / len(target_units), 2),
                           review_priority="listen_to_mismatch" if errors else "ASR_match_still_needs_quality_review",
                           asr_alignment_json=json.dumps(edits, ensure_ascii=False))
                writer.writerow(row)
                handle.flush()
                summary["processed"] += 1
                summary["asr_scored"] += 1
                summary["asr_exact_matches"] += errors == 0
                summary["asr_mismatches"] += errors > 0
                summary["reference_units"] += len(target_units)
                summary["errors"] += errors
                save_summary()
                print("  ASR mixed error rate:", row["mixed_error_rate_percent"], "%", flush=True)
        summary["status"] = "complete"
    except BaseException as exc:
        summary["status"] = "interrupted_or_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["corpus_mixed_error_rate_percent"] = (round(100 * summary["errors"] / summary["reference_units"], 2)
            if summary["reference_units"] else None)
        save_summary()
    print("\nSTEP 8A COMPLETE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Results folder:", out)
    print("Original audio and transcripts unchanged. Human review decisions are not auto-filled.")


if __name__ == "__main__":
    main()
