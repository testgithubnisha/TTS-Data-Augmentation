import os

# Use already-downloaded model files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# We already extracted speech tokens in Step 1.
os.environ.pop("onnx_path", None)

import sys
import gc
import json
import random
from pathlib import Path
from functools import partial

import torch
from transformers import AutoTokenizer
from peft import LoraConfig, inject_adapter_in_model


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR / "CosyVoice"

MODEL_DIR = (
    PROJECT_DIR / "pretrained_models" / "CosyVoice2-0.5B"
)

TRAIN_JSON = (
    PROJECT_DIR
    / "ascend_step1"
    / "train"
    / "speech_token_dataset.json"
)

sys.path.insert(0, str(REPO_DIR))

from cosyvoice.llm.llm import Qwen2Encoder, Qwen2LM
from cosyvoice.utils.common import ras_sampling


def main():
    random.seed(42)
    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this environment.")

    device = torch.device("cuda")

    print("=" * 60)
    print("STEP 2B: PRETRAINED LM TRAINING CHECK")
    print("=" * 60)
    print("Python:", sys.executable)
    print("GPU:", torch.cuda.get_device_name(0))

    # ------------------------------------------------
    # Load a complete, short training example.
    # Do not truncate audio tokens independently of text.
    # ------------------------------------------------

    with TRAIN_JSON.open("r", encoding="utf-8") as f:
        records = json.load(f)

    sample = next(
        (
            item for item in records
            if 25 <= len(item["speech_tokens"]) <= 100
            and item["answer"].strip()
        ),
        None
    )

    if sample is None:
        raise RuntimeError("No suitable short training sample found.")

    print("\nSample:", sample["utterance_id"])
    print("Transcript:", sample["answer"])
    print("Speech tokens:", len(sample["speech_tokens"]))

    # ------------------------------------------------
    # Load local text tokenizer.
    # ------------------------------------------------

    text_model_dir = MODEL_DIR / "CosyVoice-BlankEN"

    tokenizer = AutoTokenizer.from_pretrained(
        str(text_model_dir),
        local_files_only=True
    )

    text_ids = tokenizer.encode(
        sample["answer"],
        add_special_tokens=False
    )

    if not text_ids:
        raise RuntimeError("Text tokenizer returned no tokens.")

    # ------------------------------------------------
    # Construct the official CosyVoice2 language model.
    # Dimensions match the downloaded cosyvoice2.yaml.
    # ------------------------------------------------

    print("\nConstructing CosyVoice2 LM on CPU...")

    encoder = Qwen2Encoder(
        pretrain_path=str(text_model_dir)
    )

    model = Qwen2LM(
        llm_input_size=896,
        llm_output_size=896,
        speech_token_size=6561,
        llm=encoder,
        sampling=partial(
            ras_sampling,
            top_p=0.8,
            top_k=25,
            win_size=10,
            tau_r=0.1
        ),
        length_normalized_loss=True,
        lsm_weight=0.0,
        mix_ratio=[5, 15]
    )

    # Load the actual pretrained TTS weights.
    # BlankEN alone is not the complete trained speech LM.
    print("Loading pretrained llm.pt...")

    state = torch.load(
        MODEL_DIR / "llm.pt",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state, strict=True)

    del state
    gc.collect()

    print("Pretrained weights loaded with strict=True.")

    # ------------------------------------------------
    # Freeze pretrained weights and add LoRA.
    # ------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        lora_dropout=0.05,
        bias="none"
    )

    # Inject adapters without replacing the outer Qwen class.
    model.llm.model = inject_adapter_in_model(
        lora_config,
        model.llm.model
    )

    model.llm.model.config.use_cache = False

    model.llm.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("No trainable LoRA parameters found.")

    total_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_parameters)

    print("\nTotal parameters:", f"{total_count:,}")
    print("Trainable parameters:", f"{trainable_count:,}")
    print(
        "Trainable percentage:",
        f"{100 * trainable_count / total_count:.3f}%"
    )

    # ------------------------------------------------
    # Prepare batch with real text and speech token IDs.
    # ------------------------------------------------

    batch = {
        "text_token": torch.tensor(
            [text_ids],
            dtype=torch.long
        ),
        "text_token_len": torch.tensor(
            [len(text_ids)],
            dtype=torch.int32
        ),
        "speech_token": torch.tensor(
            [sample["speech_tokens"]],
            dtype=torch.long
        ),
        "speech_token_len": torch.tensor(
            [len(sample["speech_tokens"])],
            dtype=torch.int32
        )
    }

    print("Text tokens:", len(text_ids))

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=1e-4
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)

    print("\nPrecision:", amp_dtype)
    print("Running forward pass...")

    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype
    ):
        result = model(batch, device)
        loss = result["loss"]

    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Non-finite loss: {loss.item()}")

    print("Loss:", float(loss.detach().cpu()))

    print("Running backward pass...")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        max_norm=1.0
    )

    if not torch.isfinite(gradient_norm).item():
        raise RuntimeError("Non-finite LoRA gradients.")

    if gradient_norm.item() == 0:
        raise RuntimeError("LoRA gradient norm is zero.")

    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()

    print("Gradient norm:", float(gradient_norm))
    print(
        "Peak allocated GPU memory:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "GiB"
    )
    print(
        "Peak reserved GPU memory:",
        round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "GiB"
    )

    print("\n" + "=" * 60)
    print("STEP 2B PASSED")
    print("=" * 60)
    print("Pretrained CosyVoice2 LM loaded.")
    print("One LoRA optimizer step completed.")
    print("No checkpoint saved; original model files unchanged.")


if __name__ == "__main__":
    main()