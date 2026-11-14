import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.pop("onnx_path", None)

import gc
import json
import random
import hashlib
from pathlib import Path
from functools import partial

import torch
from transformers import AutoTokenizer
from peft import LoraConfig, inject_adapter_in_model
from tqdm.auto import tqdm

# Reuse the imports that already worked in Step 2B.
# Its main() does not run when imported.
from second_smoke_test import (
    Qwen2Encoder,
    Qwen2LM,
    ras_sampling,
)


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = (
    PROJECT_DIR / "pretrained_models" / "CosyVoice2-0.5B"
)

DATA_DIR = PROJECT_DIR / "ascend_step1"
OUTPUT_DIR = PROJECT_DIR / "ascend_lora_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_PATH = OUTPUT_DIR / "latest.pt"
BEST_PATH = OUTPUT_DIR / "best.pt"

EPOCHS = 3
LEARNING_RATE = 5e-5
GRAD_ACCUM = 4
MAX_SPEECH_TOKENS = 250
MAX_TEXT_TOKENS = 128
SAVE_EVERY_UPDATES = 100
SEED = 42

# Automatically resume if latest.pt exists.
RESUME = True

SETTINGS = {
    "learning_rate": LEARNING_RATE,
    "grad_accum": GRAD_ACCUM,
    "max_speech_tokens": MAX_SPEECH_TOKENS,
    "max_text_tokens": MAX_TEXT_TOKENS,
    "seed": SEED,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


# ==================================================
# DATA PREPARATION
# ==================================================

def prepare_split(split, tokenizer):
    path = DATA_DIR / split / "speech_token_dataset.json"
    raw_bytes = path.read_bytes()

    fingerprint = hashlib.sha256(raw_bytes).hexdigest()
    records = json.loads(raw_bytes.decode("utf-8"))

    prepared = []
    long_speech = 0
    long_text = 0

    for item in records:
        if item["source_split"] != split:
            raise ValueError(f"Wrong split: {item['utterance_id']}")

        speech_ids = item["speech_tokens"]

        if not speech_ids:
            raise ValueError(f"Empty speech tokens: {item['utterance_id']}")

        if any(
            type(token) is not int or not 0 <= token < 6561
            for token in speech_ids
        ):
            raise ValueError(f"Invalid tokens: {item['utterance_id']}")

        if len(speech_ids) > MAX_SPEECH_TOKENS:
            long_speech += 1
            continue

        text_ids = tokenizer.encode(
            item["answer"],
            add_special_tokens=False
        )

        if not text_ids:
            raise ValueError(f"Empty text tokens: {item['utterance_id']}")

        if len(text_ids) > MAX_TEXT_TOKENS:
            long_text += 1
            continue

        prepared.append({
            "id": item["utterance_id"],
            "text_ids": text_ids,
            "speech_ids": speech_ids,
        })

    if not prepared:
        raise RuntimeError(f"No usable samples in {split}.")

    print(
        f"{split}: kept={len(prepared)}, "
        f"long speech skipped={long_speech}, "
        f"long text skipped={long_text}"
    )

    return prepared, fingerprint


def make_batch(item):
    return {
        "text_token": torch.tensor(
            [item["text_ids"]], dtype=torch.long
        ),
        "text_token_len": torch.tensor(
            [len(item["text_ids"])], dtype=torch.int32
        ),
        "speech_token": torch.tensor(
            [item["speech_ids"]], dtype=torch.long
        ),
        "speech_token_len": torch.tensor(
            [len(item["speech_ids"])], dtype=torch.int32
        ),
    }


# ==================================================
# MODEL
# ==================================================

def build_model():
    text_model_dir = MODEL_DIR / "CosyVoice-BlankEN"

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

    state = torch.load(
        MODEL_DIR / "llm.pt",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state, strict=True)

    del state
    gc.collect()

    for parameter in model.parameters():
        parameter.requires_grad = False

    config = LoraConfig(
        r=SETTINGS["lora_r"],
        lora_alpha=SETTINGS["lora_alpha"],
        target_modules=SETTINGS["target_modules"],
        lora_dropout=SETTINGS["lora_dropout"],
        bias="none"
    )

    model.llm.model = inject_adapter_in_model(
        config,
        model.llm.model
    )

    model.llm.model.config.use_cache = False

    model.llm.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    return model


# ==================================================
# CHECKPOINTS
# ==================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    next_position,
    updates,
    best_validation,
    fingerprints
):
    # Save only trainable LoRA weights, not the entire base model.
    adapter_weights = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    checkpoint = {
        "adapter_weights": adapter_weights,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "next_position": next_position,
        "updates": updates,
        "best_validation": best_validation,
        "settings": SETTINGS,
        "fingerprints": fingerprints,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }

    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def restore_adapter(model, weights):
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if set(trainable) != set(weights):
        raise RuntimeError(
            "Checkpoint adapter names do not match this model."
        )

    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(
                weights[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype
                )
            )


# ==================================================
# VALIDATION
# ==================================================

def validate(model, records, device, amp_dtype):
    model.eval()
    saved_random_state = random.getstate()

    weighted_loss = 0.0
    target_count = 0

    try:
        with torch.inference_mode():
            for item in tqdm(records, desc="Validation"):
                # Official forward uses Python random to choose
                # streaming/non-streaming sequence layout.
                # Seed 0 selects the non-streaming branch consistently.
                random.seed(0)

                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype
                ):
                    result = model(make_batch(item), device)

                value = float(result["loss"].detach().cpu())

                if not torch.isfinite(result["loss"]).item():
                    raise RuntimeError(
                        f"Invalid validation loss: {item['id']}"
                    )

                # Speech targets plus end-of-speech target.
                count = len(item["speech_ids"]) + 1
                weighted_loss += value * count
                target_count += count

    finally:
        random.setstate(saved_random_state)
        model.train()

    return weighted_loss / target_count


# ==================================================
# TRAINING
# ==================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("GPU:", torch.cuda.get_device_name(0))
    print("Precision:", amp_dtype)
    print("Output folder:", OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR / "CosyVoice-BlankEN"),
        local_files_only=True
    )

    train_data, train_hash = prepare_split("train", tokenizer)
    validation_data, validation_hash = prepare_split(
        "validation", tokenizer
    )

    fingerprints = {
        "train": train_hash,
        "validation": validation_hash
    }

    print("\nLoading pretrained model and adding LoRA...")
    model = build_model().to(device)

    trainable = [
        p for p in model.parameters() if p.requires_grad
    ]

    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in trainable):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=not use_bf16
    )

    start_epoch = 0
    start_position = 0
    updates = 0
    best_validation = float("inf")

    if RESUME and LATEST_PATH.is_file():
        print("\nResuming:", LATEST_PATH)

        checkpoint = torch.load(
            LATEST_PATH,
            map_location="cpu",
            weights_only=True
        )

        if checkpoint["settings"] != SETTINGS:
            raise RuntimeError(
                "Training settings changed. Restore the previous "
                "settings or use a new OUTPUT_DIR for a new run."
            )

        if checkpoint["fingerprints"] != fingerprints:
            raise RuntimeError(
                "Dataset files changed since the checkpoint."
            )

        restore_adapter(model, checkpoint["adapter_weights"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = checkpoint["epoch"]
        start_position = checkpoint["next_position"]
        updates = checkpoint["updates"]
        best_validation = checkpoint["best_validation"]

        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        del checkpoint
        gc.collect()

        print("Completed optimizer updates:", updates)

    elif not RESUME and LATEST_PATH.exists():
        raise RuntimeError(
            "Existing run found. Use another OUTPUT_DIR "
            "to start a new experiment."
        )

    if start_epoch >= EPOCHS:
        print("Requested epochs already completed.")
        print("Best validation loss:", best_validation)
        return

    model.train()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, EPOCHS):
        # Reproducible shuffle, including when resuming mid-epoch.
        order = list(range(len(train_data)))
        random.Random(SEED + epoch).shuffle(order)

        position = start_position if epoch == start_epoch else 0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            total=len(order),
            initial=position,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        while position < len(order):
            # One optimizer update after this many microbatches.
            group_size = min(
                GRAD_ACCUM,
                len(order) - position
            )
            group_loss = 0.0

            for offset in range(group_size):
                item = train_data[order[position + offset]]

                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype
                ):
                    result = model(make_batch(item), device)
                    raw_loss = result["loss"]
                    loss = raw_loss / group_size

                if not torch.isfinite(raw_loss).item():
                    raise RuntimeError(
                        f"Non-finite training loss: {item['id']}"
                    )

                group_loss += float(raw_loss.detach().cpu())
                scaler.scale(loss).backward()
                progress.update(1)

                del result, raw_loss, loss

            scaler.unscale_(optimizer)

            norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=1.0
            )

            if not torch.isfinite(norm).item():
                raise RuntimeError("Non-finite training gradients.")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            position += group_size
            updates += 1

            progress.set_postfix(
                loss=f"{group_loss / group_size:.3f}",
                updates=updates
            )

            # Save only at accumulation boundaries.
            if updates % SAVE_EVERY_UPDATES == 0:
                save_checkpoint(
                    LATEST_PATH,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    position,
                    updates,
                    best_validation,
                    fingerprints
                )
                tqdm.write(f"Checkpoint saved at update {updates}")

        progress.close()

        validation_loss = validate(
            model,
            validation_data,
            device,
            amp_dtype
        )

        print(
            f"\nEpoch {epoch + 1}: "
            f"validation loss = {validation_loss:.4f}"
        )

        improved = validation_loss < best_validation

        if improved:
            best_validation = validation_loss

            save_checkpoint(
                BEST_PATH,
                model,
                optimizer,
                scaler,
                epoch + 1,
                0,
                updates,
                best_validation,
                fingerprints
            )

            print("New best checkpoint saved.")

        save_checkpoint(
            LATEST_PATH,
            model,
            optimizer,
            scaler,
            epoch + 1,
            0,
            updates,
            best_validation,
            fingerprints
        )

        print(
            "Peak reserved GPU memory:",
            round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "GiB"
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Best validation loss:", best_validation)
    print("Best checkpoint:", BEST_PATH)
    print("Latest checkpoint:", LATEST_PATH)


if __name__ == "__main__":
    main()