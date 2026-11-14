import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


# Paths are relative to this script, not the terminal directory.
PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR / "CosyVoice"

MODEL_DIR = (
    PROJECT_DIR
    / "pretrained_models"
    / "CosyVoice2-0.5B"
)

DATA_DIR = PROJECT_DIR / "ascend_step1"


def check_token_dataset(split):
    path = DATA_DIR / split / "speech_token_dataset.json"

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file missing: {path}")

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError(f"{split} dataset is empty.")

    seen_ids = set()

    for record in records:
        utterance_id = record["utterance_id"]

        if utterance_id in seen_ids:
            raise ValueError(f"Duplicate ID: {utterance_id}")

        seen_ids.add(utterance_id)

        if record.get("source_split") != split:
            raise ValueError(
                f"Wrong split metadata in {utterance_id}"
            )

        text = record.get("answer")
        tokens = record.get("speech_tokens")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty transcript: {utterance_id}")

        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Empty token sequence: {utterance_id}")

        if any(
            type(token) is not int or not 0 <= token < 6561
            for token in tokens
        ):
            raise ValueError(f"Invalid token IDs: {utterance_id}")

    print(f"{split}: {len(records)} valid records")
    return seen_ids


def main():
    print("=" * 60)
    print("Checking Existing Data")
    print("=" * 60)

    train_ids = check_token_dataset("train")
    validation_ids = check_token_dataset("validation")

    if train_ids & validation_ids:
        raise ValueError("Train and validation contain overlapping IDs.")

    # This checks IDs, not duplicate audio content or speaker overlap.
    print("Dataset structure checks passed.")

    print("\n" + "=" * 60)
    print("Checking CosyVoice Repository")
    print("=" * 60)

    required_source_files = [
        REPO_DIR / "cosyvoice" / "llm" / "llm.py",
        REPO_DIR / "third_party" / "Matcha-TTS"
        / "matcha" / "__init__.py",
    ]

    for path in required_source_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required repository file missing: {path}\n"
                "Make sure the recursive git clone completed."
            )

    print("Repository and Matcha-TTS submodule found.")

    print("\n" + "=" * 60)
    print("Checking GPU")
    print("=" * 60)

    print("PyTorch:", torch.__version__)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this Python environment."
        )

    gpu = torch.cuda.get_device_properties(0)

    print("GPU:", gpu.name)
    print("VRAM:", round(gpu.total_memory / 1024**3, 2), "GB")

    print("\n" + "=" * 60)
    print("Downloading Pretrained CosyVoice2")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="FunAudioLLM/CosyVoice2-0.5B",
        local_dir=str(MODEL_DIR),
        max_workers=2,
    )

    required_model_files = [
        "cosyvoice2.yaml",
        "llm.pt",
        "flow.pt",
        "hift.pt",
        "campplus.onnx",
        "speech_tokenizer_v2.onnx",
        "CosyVoice-BlankEN/config.json",
        "CosyVoice-BlankEN/model.safetensors",
        "CosyVoice-BlankEN/tokenizer_config.json",
        "CosyVoice-BlankEN/vocab.json",
        "CosyVoice-BlankEN/merges.txt",
    ]

    missing = [
        name
        for name in required_model_files
        if not (MODEL_DIR / name).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Missing model files:\n" + "\n".join(missing)
        )

    print("\n" + "=" * 60)
    print("STEP 2A COMPLETE")
    print("=" * 60)

    print("Repository:", REPO_DIR)
    print("Pretrained model:", MODEL_DIR)
    print("Train and validation files checked.")
    print("Required model files found.")
    print("Model loading and training have not started yet.")


if __name__ == "__main__":
    main()