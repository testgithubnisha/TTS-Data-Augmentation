"""Step 5: 20 paired listening examples using the working fourth_generate.py.

Place this file beside fourth_generate.py and run with .venv_cosyvoice.
This is an exploratory listening check, not a scored benchmark.
"""

import csv
import gc
import hashlib
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if not (PROJECT_DIR / "fourth_generate.py").is_file():
    raise FileNotFoundError("Place this file beside your working fourth_generate.py.")

# Its main() guard prevents Step 4 from running on import.
import fourth_generate as step4

PROMPTS = [
    ("short", "你好，good morning。", "Hello, good morning."),
    ("short", "谢谢你的 help。", "Thank you for your help."),
    ("short", "我想喝一杯 coffee。", "I would like a cup of coffee."),
    ("short", "请打开这个 file。", "Please open this file."),
    ("short", "这个 idea 很有意思。", "This idea is very interesting."),
    ("medium", "我今天下午有一个 meeting。", "I have a meeting this afternoon."),
    ("medium", "你可以把这个 document 发给我吗？", "Can you send me this document?"),
    ("medium", "这个 project 比我想象的难。", "This project is harder than I expected."),
    ("medium", "我已经完成了今天的 homework。", "I have finished today's homework."),
    ("medium", "我们先检查一下这个 result。", "Let us check this result first."),
    ("english_first", "Please wait，我马上回来。", "Please wait; I will be right back."),
    ("english_first", "Good morning，今天我们讨论实验结果。", "Good morning; today we discuss the experimental results."),
    ("english_first", "Can you help me 检查这个句子？", "Can you help me check this sentence?"),
    ("english_first", "Let us start，大家准备好了吗？", "Let us start; is everyone ready?"),
    ("english_first", "Thank you，这个建议很有用。", "Thank you; this suggestion is very useful."),
    ("multiple_switches", "我打开了 laptop，然后开始写 report。", "I opened my laptop and then started writing the report."),
    ("multiple_switches", "这个 model 的声音很清楚，但是 speed 有点快。", "This model's voice is clear, but the speed is a little fast."),
    ("multiple_switches", "请先检查 audio，再修改对应的 text。", "Please check the audio first, then edit the corresponding text."),
    ("multiple_switches", "我们明天有一个 meeting，讨论新的 project plan。", "We have a meeting tomorrow to discuss the new project plan."),
    ("multiple_switches", "如果这个 experiment 成功，我们就可以准备 presentation。", "If this experiment succeeds, we can prepare the presentation."),
]

FIELDS = [
    "sample_id", "category", "text", "meaning_en", "model", "seed",
    "wav_path", "duration_seconds", "sample_rate", "status", "error",
    "all_words_correct_yes_no_unsure", "missing_or_extra_words",
    "repetition_yes_no", "pronunciation_1_to_5", "naturalness_1_to_5", "notes",
]


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if c.isalnum())


def check_overlap():
    """Check exact normalized text overlap only, not semantic similarity."""
    report = {}
    for split in ("train", "validation"):
        path = PROJECT_DIR / "ascend_step1" / split / "speech_token_dataset.json"
        if not path.is_file():
            report[split] = {"checked": False, "reason": "JSON not found"}
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        texts = {normalized(item["answer"]) for item in records}
        matches = [f"sample_{i:03d}" for i, (_, text, _) in enumerate(PROMPTS, 1)
                   if normalized(text) in texts]
        report[split] = {"checked": True, "matching_sample_ids": matches}
        print(f"{split}: {len(matches)} exact normalized prompt matches")
    return report


def main():
    if not step4.torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. Use your .venv_cosyvoice interpreter.")
    for path in (step4.CHECKPOINT_PATH, step4.REFERENCE_WAV,
                 step4.MODEL_DIR / "cosyvoice2.yaml"):
        if not path.is_file():
            raise FileNotFoundError(path)

    # A new directory for every run preserves all previous audio and ratings.
    run_dir = PROJECT_DIR / "batch_comparison" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    for variant in ("baseline", "finetuned"):
        (run_dir / variant).mkdir()

    print("=" * 60)
    print("STEP 5: 20 SENTENCES, 40 AUDIO FILES")
    print("Python:", sys.executable)
    print("GPU:", step4.torch.cuda.get_device_name(0))
    print("Output:", run_dir)
    overlap = check_overlap()
    metadata = {
        "purpose": "Exploratory paired listening check; not a held-out benchmark",
        "checkpoint": str(step4.CHECKPOINT_PATH),
        "reference_wav": str(step4.REFERENCE_WAV),
        "reference_text": step4.REFERENCE_TEXT,
        "fourth_generate_sha256": hashlib.sha256(
            (PROJECT_DIR / "fourth_generate.py").read_bytes()).hexdigest(),
        "text_overlap_check": overlap,
        "seed_policy": "42 + zero-based sample index, same seed in each pair; GPU execution may vary",
        "prompts": [{"sample_id": f"sample_{i:03d}", "category": category,
                     "text": text, "meaning_en": meaning}
                    for i, (category, text, meaning) in enumerate(PROMPTS, 1)],
        "status": "started",
    }
    metadata_path = run_dir / "run_info.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    save_metadata()
    print("Loading the same inference components used in Step 4...")
    configs = step4.load_inference_config()
    sample_rate = int(configs["sample_rate"])
    frontend = step4.CosyVoiceFrontEnd(
        get_tokenizer=configs["get_tokenizer"],
        feat_extractor=configs["feat_extractor"],
        campplus_model=str(step4.MODEL_DIR / "campplus.onnx"),
        speech_tokenizer_model=str(step4.MODEL_DIR / "speech_tokenizer_v2.onnx"),
        allowed_special=configs["allowed_special"],
    )
    model = step4.CosyVoice2Model(configs["llm"], configs["flow"], configs["hift"], fp16=True)
    model.load(str(step4.MODEL_DIR / "llm.pt"), str(step4.MODEL_DIR / "flow.pt"),
               str(step4.MODEL_DIR / "hift.pt"))
    del configs
    gc.collect()
    metadata["sample_rate"] = sample_rate
    save_metadata()

    csv_path = run_dir / "listening_comparison.csv"
    completed = 0
    try:
        # Flush every row so successful results remain documented if interrupted.
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            handle.flush()
            for variant in ("baseline", "finetuned"):
                if variant == "finetuned":
                    print("\nApplying best.pt once for all fine-tuned examples...")
                    step4.apply_trained_lora(model)
                step4.OUTPUT_DIR = run_dir / variant
                for index, (category, text, meaning) in enumerate(PROMPTS, 1):
                    sample_id = f"sample_{index:03d}"
                    step4.TARGET_TEXT = text
                    step4.SEED = 42 + index - 1
                    row = dict(sample_id=sample_id, category=category, text=text,
                               meaning_en=meaning, model=variant, seed=step4.SEED,
                               sample_rate=sample_rate)
                    print(f"\n[{completed + 1}/40] {variant} / {sample_id}: {text}")
                    try:
                        path = step4.generate_audio(frontend, model, sample_rate, f"{sample_id}.wav")
                        row.update(wav_path=str(path.resolve()),
                                   duration_seconds=round(step4.sf.info(str(path)).duration, 4),
                                   status="generated")
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        writer.writerow(row)
                        handle.flush()
                        raise
                    writer.writerow(row)
                    handle.flush()
                    completed += 1
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "interrupted_or_failed"
        raise
    finally:
        metadata["completed_audio_files"] = completed
        save_metadata()

    print("\n" + "=" * 60)
    print("STEP 5 GENERATION COMPLETE")
    print("Audio files:", completed)
    print("CSV:", csv_path)
    print("Compare matching sample IDs in baseline and finetuned folders.")
    print("Fill blank listening columns manually; 1 = poor, 5 = excellent.")
    print("Use 'unsure' for words/pronunciation you cannot judge.")
    print("Duration is descriptive, not a quality score.")


if __name__ == "__main__":
    main()
