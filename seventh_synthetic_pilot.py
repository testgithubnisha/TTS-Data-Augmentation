"""Generate up to 100 synthetic training examples from ASCEND train only.
Uses fourth_generate.py and the existing best.pt. No model training occurs.
Single reference speaker pilot; not a full reproduction of the paper.
"""
import argparse
import gc
import hashlib
import json
import random
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def select_records(count):
    train_path = ROOT / "ascend_step1/train/speech_token_dataset.json"
    val_path = ROOT / "ascend_step1/validation/speech_token_dataset.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    validation = json.loads(val_path.read_text(encoding="utf-8"))
    # Validation is used only to exclude overlapping text, never to synthesize it.
    excluded = {normalized(r["answer"]) for r in validation}
    seen = set()
    pool = []
    for index, record in enumerate(train):
        split = record.get("source_split")
        if split is not None and split != "train":
            raise ValueError(f"Non-train record in training JSON at index {index}: {split}")
        text = record["answer"].strip()
        if not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text):
            continue
        if not re.search(r"[A-Za-z]{2,}", text):
            continue
        # Keep the first pilot short and avoid obvious transcript annotations.
        if not 5 <= len(text) <= 120:
            continue
        if any(char in text for char in "[]<>{}"):
            continue
        if not 25 <= len(record.get("speech_tokens", [])) <= 250:
            continue
        key = normalized(text)
        if key in excluded or key in seen:
            continue
        seen.add(key)
        pool.append({"source_id": record.get("utterance_id", f"ascend_train_{index:06d}"),
                     "source_index": record.get("source_index", index),
                     "text": text})
    random.Random(42).shuffle(pool)
    if not pool:
        raise ValueError("No eligible mixed-language training examples found.")
    return pool[:count], len(pool), train_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("For this pilot, --count must be between 1 and 100.")
    print("STEP 7: SYNTHETIC TRAINING PILOT", flush=True)
    selected, eligible, train_path = select_records(args.count)
    print(f"Eligible unique texts: {eligible}; selected: {len(selected)}", flush=True)
    print("Loading existing Step 4 components...", flush=True)
    import fourth_generate as s4
    if not s4.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run with .venv_cosyvoice.")
    for path in (s4.CHECKPOINT_PATH, s4.REFERENCE_WAV):
        if not path.is_file():
            raise FileNotFoundError(path)
    configs = s4.load_inference_config()
    sr = int(configs["sample_rate"])
    frontend = s4.CosyVoiceFrontEnd(
        get_tokenizer=configs["get_tokenizer"], feat_extractor=configs["feat_extractor"],
        campplus_model=str(s4.MODEL_DIR / "campplus.onnx"),
        speech_tokenizer_model=str(s4.MODEL_DIR / "speech_tokenizer_v2.onnx"),
        allowed_special=configs["allowed_special"])
    model = s4.CosyVoice2Model(configs["llm"], configs["flow"], configs["hift"], fp16=True)
    model.load(str(s4.MODEL_DIR / "llm.pt"), str(s4.MODEL_DIR / "flow.pt"),
               str(s4.MODEL_DIR / "hift.pt"))
    del configs
    gc.collect()
    s4.apply_trained_lora(model)

    out = ROOT / "synthetic_pilot" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    (out / "audio").mkdir(parents=True, exist_ok=False)
    s4.OUTPUT_DIR = out / "audio"
    print("Python:", sys.executable)
    print("GPU:", s4.torch.cuda.get_device_name(0))
    print("Output:", out)
    metadata = {
        "status": "running", "requested": args.count, "selected": len(selected),
        "eligible": eligible, "completed": 0, "selection_seed": 42,
        "source_dataset": "CAiRE/ASCEND", "source_split": "train",
        "train_json_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "checkpoint": str(s4.CHECKPOINT_PATH), "reference_wav": str(s4.REFERENCE_WAV),
        "reference_text": s4.REFERENCE_TEXT, "sample_rate": sr,
        "validation_use": "Exclude exact normalized text overlap only",
        "selection": "Han character + Latin word >=2 letters; 5-120 text chars; 25-250 source speech tokens; no []<>{} annotations; unique NFKC lowercase alphanumeric text",
        "purpose": "Single reference speaker training pilot; generated audio is not automatically quality-approved",
        "asr_training_sample_rate": "Resample in the later ASR preprocessing step; WAV files remain at native TTS sample rate",
    }
    (out / "selected_texts.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    def save_progress():
        (out / "run_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    save_progress()
    try:
        with (out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for i, record in enumerate(selected, 1):
                sample_id = f"synthetic_{i:04d}"
                s4.TARGET_TEXT = record["text"]
                s4.SEED = 1000 + i
                print(f"\n[{i}/{len(selected)}] {sample_id}: {record['text']}", flush=True)
                path = s4.generate_audio(frontend, model, sr, sample_id + ".wav")
                info = s4.sf.info(str(path))
                item = dict(record, synthetic_id=sample_id,
                            audio_path=path.relative_to(out).as_posix(),
                            source_dataset="CAiRE/ASCEND", source_split="train",
                            sample_rate=info.samplerate, duration_seconds=info.duration,
                            generation_seed=s4.SEED, is_synthetic=True,
                            reference_speaker="repository_zero_shot_prompt",
                            quality_status="pending_review")
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                metadata["completed"] = i
                save_progress()
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata["status"] = "interrupted_or_failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_progress()
    print("\nSTEP 7 GENERATION COMPLETE")
    print("Generated files:", metadata["completed"])
    print("Manifest:", out / "manifest.jsonl")
    print("Run info:", out / "run_info.json")
    print("These are training candidates. Review quality before ASR training.")


if __name__ == "__main__":
    main()
