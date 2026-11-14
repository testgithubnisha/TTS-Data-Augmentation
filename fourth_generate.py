import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.pop("onnx_path", None)

import sys
import gc
import json
import random
from pathlib import Path

import yaml
import numpy as np
import torch
import soundfile as sf

from hyperpyyaml import load_hyperpyyaml
from peft import LoraConfig, inject_adapter_in_model


# ==================================================
# PATHS AND INPUT TEXT
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR / "CosyVoice"

MODEL_DIR = (
    PROJECT_DIR / "pretrained_models" / "CosyVoice2-0.5B"
)

CHECKPOINT_PATH = (
    PROJECT_DIR / "ascend_lora_training" / "best.pt"
)

OUTPUT_DIR = PROJECT_DIR / "generated_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_WAV = REPO_DIR / "asset" / "zero_shot_prompt.wav"

# Exact transcript paired with the repository's reference WAV.
REFERENCE_TEXT = "希望你以后能够做的比我还好呦。"

# Start with the text from your original test.
TARGET_TEXT = "hello hello 可以"

SEED = 42

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice2Model


# ==================================================
# INFERENCE CONFIGURATION
# ==================================================

def load_inference_config():
    config_path = MODEL_DIR / "cosyvoice2.yaml"

    # Preserve HyperPyYAML tags/references while excluding unused
    # training data processors and GAN training objects.
    document = yaml.compose(
        config_path.read_text(encoding="utf-8")
    )

    keep_keys = {
        "sample_rate",
        "llm_input_size",
        "llm_output_size",
        "spk_embed_dim",
        "qwen_pretrain_path",
        "token_frame_rate",
        "token_mel_ratio",
        "chunk_size",
        "num_decoding_left_chunks",
        "llm",
        "flow",
        "hift",
        "get_tokenizer",
        "token_path",
        "allowed_special",
        "feat_extractor",
    }

    document.value = [
        (key, value)
        for key, value in document.value
        if key.value in keep_keys
    ]

    inference_yaml = yaml.serialize(document)

    return load_hyperpyyaml(
        inference_yaml,
        overrides={
            "qwen_pretrain_path": str(
                MODEL_DIR / "CosyVoice-BlankEN"
            )
        }
    )


# ==================================================
# REPRODUCIBILITY
# ==================================================

def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ==================================================
# AUDIO GENERATION
# ==================================================

def generate_audio(frontend, model, sample_rate, filename):
    reset_seed()

    model.llm.eval()
    model.flow.eval()
    model.hift.eval()

    # Direct frontend call uses the supplied text as-is.
    # No optional text-normalization package is needed.
    model_inputs = frontend.frontend_zero_shot(
        TARGET_TEXT,
        REFERENCE_TEXT,
        str(REFERENCE_WAV),
        sample_rate,
        ""
    )

    chunks = []

    with torch.inference_mode():
        for result in model.tts(
            **model_inputs,
            stream=False,
            speed=1.0
        ):
            chunks.append(
                result["tts_speech"].detach().float().cpu()
            )

    if not chunks:
        raise RuntimeError(f"No audio returned for {filename}.")

    waveform = torch.cat(chunks, dim=-1).squeeze(0).numpy()

    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError("Generated waveform is empty or non-finite.")

    peak = float(np.abs(waveform).max())

    if peak == 0:
        raise RuntimeError("Generated waveform is entirely silent.")

    # Avoid clipping when saving PCM audio.
    if peak > 1.0:
        waveform = waveform / peak * 0.99

    output_path = OUTPUT_DIR / filename

    sf.write(
        str(output_path),
        waveform,
        sample_rate,
        subtype="PCM_16"
    )

    print("Saved:", output_path)
    print("Duration:", round(len(waveform) / sample_rate, 2), "seconds")
    print("Peak amplitude before saving:", round(peak, 4))

    return output_path


# ==================================================
# LOAD CUSTOM TRAINING CHECKPOINT
# ==================================================

def apply_trained_lora(model):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    settings = checkpoint["settings"]

    config = LoraConfig(
        r=settings["lora_r"],
        lora_alpha=settings["lora_alpha"],
        target_modules=settings["target_modules"],
        lora_dropout=settings["lora_dropout"],
        bias="none"
    )

    # Same injection location used during training.
    model.llm.llm.model = inject_adapter_in_model(
        config,
        model.llm.llm.model
    )

    saved_weights = checkpoint["adapter_weights"]

    # Training saved names relative to the outer Qwen2LM.
    adapter_parameters = {
        name: parameter
        for name, parameter in model.llm.named_parameters()
        if "lora_" in name
    }

    missing = set(adapter_parameters) - set(saved_weights)
    unexpected = set(saved_weights) - set(adapter_parameters)

    if missing or unexpected:
        raise RuntimeError(
            "LoRA checkpoint/model mismatch.\n"
            f"Missing names: {sorted(missing)[:5]}\n"
            f"Unexpected names: {sorted(unexpected)[:5]}"
        )

    with torch.no_grad():
        for name, parameter in adapter_parameters.items():
            saved = saved_weights[name]

            if saved.shape != parameter.shape:
                raise RuntimeError(f"Shape mismatch: {name}")

            parameter.copy_(
                saved.to(
                    device=parameter.device,
                    dtype=parameter.dtype
                )
            )

    # Keep inference precision consistent with the base LM.
    model.llm.half()
    model.llm.eval()

    for parameter in model.llm.parameters():
        parameter.requires_grad = False

    print("Loaded LoRA tensors:", len(adapter_parameters))
    print("Best validation loss:", checkpoint["best_validation"])

    del checkpoint
    gc.collect()


# ==================================================
# MAIN
# ==================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable.")

    for path in [
        MODEL_DIR / "cosyvoice2.yaml",
        CHECKPOINT_PATH,
        REFERENCE_WAV,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    print("=" * 60)
    print("STEP 4: BASELINE VS FINE-TUNED AUDIO")
    print("=" * 60)

    print("Python:", sys.executable)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Target text:", TARGET_TEXT)
    print("Reference audio:", REFERENCE_WAV)

    # Verify that the reference is a readable audio file.
    reference_info = sf.info(str(REFERENCE_WAV))
    print(
        "Reference duration:",
        round(reference_info.duration, 2),
        "seconds"
    )

    print("\nLoading inference components...")
    configs = load_inference_config()

    sample_rate = int(configs["sample_rate"])

    frontend = CosyVoiceFrontEnd(
        get_tokenizer=configs["get_tokenizer"],
        feat_extractor=configs["feat_extractor"],
        campplus_model=str(MODEL_DIR / "campplus.onnx"),
        speech_tokenizer_model=str(
            MODEL_DIR / "speech_tokenizer_v2.onnx"
        ),
        allowed_special=configs["allowed_special"]
    )

    model = CosyVoice2Model(
        configs["llm"],
        configs["flow"],
        configs["hift"],
        fp16=True
    )

    model.load(
        str(MODEL_DIR / "llm.pt"),
        str(MODEL_DIR / "flow.pt"),
        str(MODEL_DIR / "hift.pt")
    )

    del configs
    gc.collect()

    print("Output sample rate:", sample_rate)

    print("\nGenerating BASELINE audio...")
    generate_audio(
        frontend,
        model,
        sample_rate,
        "baseline.wav"
    )

    print("\nApplying trained LoRA checkpoint...")
    apply_trained_lora(model)

    print("\nGenerating FINE-TUNED audio...")
    generate_audio(
        frontend,
        model,
        sample_rate,
        "finetuned.wav"
    )

    # Save the inputs used for this comparison.
    with (OUTPUT_DIR / "comparison_info.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "target_text": TARGET_TEXT,
                "reference_text": REFERENCE_TEXT,
                "reference_wav": str(REFERENCE_WAV),
                "checkpoint": str(CHECKPOINT_PATH),
                "sample_rate": sample_rate,
                "seed": SEED
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE")
    print("=" * 60)
    print("Listen to:")
    print(OUTPUT_DIR / "baseline.wav")
    print(OUTPUT_DIR / "finetuned.wav")


if __name__ == "__main__":
    main()