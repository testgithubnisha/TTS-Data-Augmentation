"""Exploratory ASR comparison; does not supply human listening ratings.
Run beside batch_comparison. Existing WAV and listening CSV files stay intact.
"""
import argparse
import csv
import json
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path


def units(text):
    # NFKC, lowercase, ignore punctuation; Han characters + English words.
    # Traditional/simplified variants remain different. No translation mapping.
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z]+(?:'[a-z]+)*|\d+", text)


def align(reference, hypothesis):
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(costs[i-1][j] + 1, costs[i][j-1] + 1,
                              costs[i-1][j-1] + (reference[i-1] != hypothesis[j-1]))
    i, j = n, m
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    edits = []
    while i or j:
        if i and j and reference[i-1] == hypothesis[j-1] and costs[i][j] == costs[i-1][j-1]:
            i, j = i-1, j-1
        elif i and j and costs[i][j] == costs[i-1][j-1] + 1:
            counts["substitutions"] += 1
            edits.append({"op": "substitution", "target": reference[i-1], "asr": hypothesis[j-1]})
            i, j = i-1, j-1
        elif i and costs[i][j] == costs[i-1][j] + 1:
            counts["deletions"] += 1
            edits.append({"op": "deletion", "target": reference[i-1]})
            i -= 1
        else:
            counts["insertions"] += 1
            edits.append({"op": "insertion", "asr": hypothesis[j-1]})
            j -= 1
    return counts, list(reversed(edits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--model", choices=["small", "medium"], default="small")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted((project / "batch_comparison").glob("run_*/listening_comparison.csv"))
        if not candidates:
            raise FileNotFoundError("No listening_comparison.csv found; pass --run-dir PATH.")
        run_dir = candidates[-1].parent
    run_dir = run_dir.resolve()
    with (run_dir / "listening_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("The listening CSV is empty.")
    seen = set()
    for row in rows:
        key = (row["model"], row["sample_id"])
        if key in seen or row["model"] not in {"baseline", "finetuned"}:
            raise ValueError(f"Unexpected/duplicate row: {key}")
        seen.add(key)
        if not units(row["text"]):
            raise ValueError(f"Empty normalized target: {key}")
        # Resolve relative to the run folder, not the old Windows CSV paths.
        path = run_dir / row["model"] / (row["sample_id"] + ".wav")
        if not path.is_file():
            raise FileNotFoundError(path)
    for _, sample_id in seen:
        if not all((v, sample_id) in seen for v in ("baseline", "finetuned")):
            raise ValueError(f"Missing comparison partner for {sample_id}")

    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    from scipy.signal import resample_poly

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("STEP 6: AUTOMATIC ASR CHECK (NOT HUMAN QUALITY RATINGS)")
    print("Input:", run_dir)
    print("Files:", len(rows), "| Whisper:", args.model, "| Device:", device)
    print("First use downloads Whisper weights; TTS weights/checkpoints are not changed.")
    model = whisper.load_model(args.model, device=device,
                               download_root=str(project / "whisper_models"))
    model.eval()
    out = run_dir / datetime.now().strftime("asr_%Y%m%d_%H%M%S_%f")
    out.mkdir(exist_ok=False)
    csv_path = out / "automatic_comparison.csv"
    fields = ["sample_id", "model", "target_text", "asr_text", "asr_model",
              "reference_units", "substitutions", "deletions", "insertions",
              "mixed_error_rate_percent", "normalized_exact_match", "review_flag",
              "asr_alignment_json", "wav_path"]
    totals = {v: dict(files=0, exact_matches=0, reference_units=0, errors=0)
              for v in ("baseline", "finetuned")}
    settings = dict(asr_model=args.model, device=device, language="zh", task="transcribe",
                    temperature=0, beam_size=1, condition_on_previous_text=False,
                    target_text_given_to_asr=False,
                    normalization="NFKC lowercase; Han character/English word/digit-run units; punctuation ignored; no script conversion",
                    limitation="ASR mismatch may be recognition error, not TTS error. No pronunciation/naturalness scores. Exploratory prompts only.")
    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()
        for index, row in enumerate(rows, 1):
            path = run_dir / row["model"] / (row["sample_id"] + ".wav")
            samples, sr = sf.read(path, dtype="float32", always_2d=True)
            samples = samples.mean(axis=1)
            if not samples.size or not np.isfinite(samples).all():
                raise ValueError(f"Invalid waveform: {path}")
            if sr != 16000:
                divisor = math.gcd(sr, 16000)
                samples = resample_poly(samples, 16000 // divisor, sr // divisor)
            samples = np.ascontiguousarray(samples, dtype=np.float32)
            # No target text or previous transcript is supplied to the recognizer.
            with torch.inference_mode():
                result = model.transcribe(samples, language="zh", task="transcribe",
                    temperature=0.0, beam_size=1, condition_on_previous_text=False,
                    fp16=(device == "cuda"), verbose=None)
            hypothesis = result["text"].strip()
            reference_units, hypothesis_units = units(row["text"]), units(hypothesis)
            counts, edits = align(reference_units, hypothesis_units)
            errors = sum(counts.values())
            output = dict(sample_id=row["sample_id"], model=row["model"],
                target_text=row["text"], asr_text=hypothesis, asr_model=args.model,
                reference_units=len(reference_units), **counts,
                mixed_error_rate_percent=round(100 * errors / len(reference_units), 2),
                normalized_exact_match=(errors == 0),
                review_flag="ASR mismatch: listen to verify" if errors else "ASR matches; listening quality not assessed",
                asr_alignment_json=json.dumps(edits, ensure_ascii=False), wav_path=str(path))
            writer.writerow(output)
            f.flush()
            total = totals[row["model"]]
            total["files"] += 1
            total["reference_units"] += len(reference_units)
            total["errors"] += errors
            total["exact_matches"] += int(errors == 0)
            print(f"\n[{index}/{len(rows)}] {row['model']} {row['sample_id']}")
            print("Target:", row["text"])
            print("ASR:   ", hypothesis)
            print("Mixed error rate:", output["mixed_error_rate_percent"], "%")
    for value in totals.values():
        value["corpus_mixed_error_rate_percent"] = round(100 * value["errors"] / value["reference_units"], 2)
    summary = dict(settings=settings, results=totals)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSTEP 6 COMPLETE")
    print(json.dumps(totals, indent=2))
    print("CSV:", csv_path)
    print("Summary:", out / "summary.json")
    print("Lower ASR disagreement is a proxy, not proof of better pronunciation or naturalness.")


if __name__ == "__main__":
    main()
