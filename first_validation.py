import io
import os
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import soundfile as sf
import whisper
import onnxruntime as ort

from scipy.signal import resample_poly
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm


# ==================================================
# CONFIGURATION
# ==================================================

DATASET_NAME = "CAiRE/ASCEND"
SPLIT = "validation"

MODEL_REPO = "FunAudioLLM/CosyVoice2-0.5B"
MODEL_FILE = "speech_tokenizer_v2.onnx"

TARGET_SR = 16000
VOCAB_SIZE = 6561
MAX_AUDIO_SECONDS = 30

# Separate folder to avoid mixing with old invalid tokens.
OUTPUT_DIR = Path("ascend_step1") / SPLIT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "speech_token_dataset.json"
ERROR_JSON = OUTPUT_DIR / "skipped_samples.json"
TOKEN_PT = OUTPUT_DIR / "utt2speech_token.pt"

# None = process entire dataset.
# Set to 10 for a quick initial test.
MAX_SAMPLES = None

SAVE_EVERY = 100


# ==================================================
# SAVE HELPERS
# ==================================================

def save_json_atomic(data, path):
    """Write to a temporary file, then replace the destination."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(temporary_path, path)


# ==================================================
# DOWNLOAD AND LOAD REAL SPEECH TOKENIZER
# ==================================================

print("=" * 60)
print("Downloading CosyVoice2 Speech Tokenizer")
print("=" * 60)

tokenizer_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

options = ort.SessionOptions()
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)
options.intra_op_num_threads = min(4, os.cpu_count() or 1)

session = ort.InferenceSession(
    tokenizer_path,
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

model_inputs = session.get_inputs()

if len(model_inputs) != 2:
    raise RuntimeError(
        f"Expected 2 tokenizer inputs, found {len(model_inputs)}."
    )

feature_input = model_inputs[0]
length_input = model_inputs[1]

if feature_input.type != "tensor(float)":
    raise RuntimeError(
        f"Unexpected feature input type: {feature_input.type}"
    )

length_dtypes = {
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}

if length_input.type not in length_dtypes:
    raise RuntimeError(
        f"Unexpected length input type: {length_input.type}"
    )

length_dtype = length_dtypes[length_input.type]

print("Tokenizer loaded.")
print("Provider:", session.get_providers())

for item in model_inputs:
    print(
        "Input:",
        item.name,
        "| type:", item.type,
        "| shape:", item.shape
    )


# ==================================================
# LOAD DATASET
# ==================================================

print("\n" + "=" * 60)
print("Loading Dataset")
print("=" * 60)

dataset = load_dataset(
    DATASET_NAME,
    split=SPLIT
)

dataset = dataset.rename_columns({
    "audio": "context",
    "transcription": "answer"
})
dataset = dataset.cast_column(
    "context",
    Audio(decode=False)
)

required_columns = {"context", "answer"}

if not required_columns.issubset(dataset.column_names):
    raise RuntimeError(
        f"Missing required columns. Found: {dataset.column_names}"
    )

number_to_process = len(dataset)

if MAX_SAMPLES is not None:
    number_to_process = min(MAX_SAMPLES, number_to_process)

print("Available samples:", len(dataset))
print("Samples to process:", number_to_process)
print("Split:", SPLIT)


# ==================================================
# AUDIO LOADING AND PREPROCESSING
# ==================================================

def read_audio(audio_info):
    """Read embedded audio bytes or an available local audio path."""
    audio_bytes = audio_info.get("bytes")
    audio_path = audio_info.get("path")

    if audio_bytes is not None:
        source = io.BytesIO(audio_bytes)
    elif audio_path and Path(audio_path).is_file():
        source = audio_path
    else:
        raise ValueError("No readable audio bytes or local path.")

    # SoundFile returns [samples, channels].
    waveform, original_sr = sf.read(
        source,
        dtype="float32",
        always_2d=True
    )

    if waveform.shape[0] == 0 or original_sr <= 0:
        raise ValueError("Empty audio or invalid sample rate.")

    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains NaN or infinity.")

    duration = waveform.shape[0] / original_sr

    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio is {duration:.2f}s; tokenizer limit is "
            f"{MAX_AUDIO_SECONDS}s. Segment with aligned text first."
        )

    # Stereo/multichannel -> mono.
    waveform = waveform.mean(axis=1)

    if original_sr != TARGET_SR:
        common = math.gcd(int(original_sr), TARGET_SR)

        waveform = resample_poly(
            waveform,
            up=TARGET_SR // common,
            down=int(original_sr) // common
        )

    waveform = np.ascontiguousarray(
        waveform,
        dtype=np.float32
    )

    # Whisper's centered STFT needs sufficient input length.
    if waveform.size < 400:
        raise ValueError("Audio is too short for feature extraction.")

    return waveform, int(original_sr)


# ==================================================
# EXTRACT DISCRETE COSYVOICE2 TOKEN IDS
# ==================================================

def extract_tokens(waveform):
    audio_tensor = torch.from_numpy(waveform).unsqueeze(0)

    with torch.inference_mode():
        features = whisper.log_mel_spectrogram(
            audio_tensor,
            n_mels=128
        )

    features_np = np.ascontiguousarray(
        features.cpu().numpy(),
        dtype=np.float32
    )

    feature_length = np.array(
        [features_np.shape[-1]],
        dtype=length_dtype
    )

    result = session.run(
        None,
        {
            feature_input.name: features_np,
            length_input.name: feature_length
        }
    )

    token_array = np.asarray(result[0]).reshape(-1)

    if token_array.size == 0:
        raise RuntimeError("Tokenizer returned an empty sequence.")

    if not np.issubdtype(token_array.dtype, np.integer):
        raise RuntimeError(
            f"Expected integer token IDs, got {token_array.dtype}."
        )

    if token_array.min() < 0 or token_array.max() >= VOCAB_SIZE:
        raise RuntimeError(
            f"Token IDs outside expected range 0–{VOCAB_SIZE - 1}: "
            f"min={token_array.min()}, max={token_array.max()}"
        )

    return token_array.tolist()


# ==================================================
# PROCESS DATASET
# ==================================================

print("\n" + "=" * 60)
print("Extracting CosyVoice2 Speech Tokens")
print("=" * 60)

records = []
skipped = []

try:
    for idx in tqdm(range(number_to_process)):
        # Stable ID preserves alignment if some samples are skipped.
        utterance_id =  f"ascend_{SPLIT}_{idx:06d}"

        try:
            sample = dataset[idx]

            transcript = sample["answer"]

            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError("Missing or empty transcript.")

            waveform, original_sr = read_audio(sample["context"])

        except Exception as error:
            skipped.append({
                "utterance_id": utterance_id,
                "source_index": idx,
                "error": str(error)
            })

            tqdm.write(f"Skipped {utterance_id}: {error}")
            continue

        # Fail visibly on tokenizer/model errors instead of silently
        # skipping every sample due to a broken installation.
        speech_tokens = extract_tokens(waveform)

        records.append({
            "original_id": sample["id"],
            "speaker_id": sample["original_speaker_id"],
            "session_id": sample["session_id"],
            "language": sample["language"],
            "topic": sample["topic"],
            "utterance_id": utterance_id,
            "source_dataset": DATASET_NAME,
            "source_split": SPLIT,
            "source_index": idx,
            "instruction": sample.get("instruction", ""),
            "answer": transcript,
            "speech_tokens": speech_tokens,
            "num_speech_tokens": len(speech_tokens),
            "duration_seconds": len(waveform) / TARGET_SR,
            "original_sample_rate": original_sr,
            "tokenizer_sample_rate": TARGET_SR,
            "speech_tokenizer": f"{MODEL_REPO}/{MODEL_FILE}"
        })

        if len(records) == 1:
            tqdm.write(
                f"First sample: {len(speech_tokens)} tokens, "
                f"{len(set(speech_tokens))} unique\n"
                f"First 20: {speech_tokens[:20]}"
            )

        if len(records) % SAVE_EVERY == 0:
            save_json_atomic(records, OUTPUT_JSON)
            save_json_atomic(skipped, ERROR_JSON)

finally:
    # Preserve completed work even if interrupted.
    save_json_atomic(records, OUTPUT_JSON)
    save_json_atomic(skipped, ERROR_JSON)


# ==================================================
# SAVE ID -> TOKEN MAPPING
# ==================================================

if not records:
    raise RuntimeError(
        f"No samples processed successfully. Check {ERROR_JSON}."
    )

utterance_to_tokens = {
    record["utterance_id"]: record["speech_tokens"]
    for record in records
}

torch.save(utterance_to_tokens, TOKEN_PT)


# ==================================================
# VALIDATION SUMMARY
# ==================================================

counts = Counter(
    token
    for record in records
    for token in record["speech_tokens"]
)

total_tokens = sum(counts.values())
most_common_token, most_common_count = counts.most_common(1)[0]

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)

print("Processed samples:", len(records))
print("Skipped samples:", len(skipped))
print("Total speech tokens:", total_tokens)
print("Unique speech tokens:", len(counts))
print("Token range:", min(counts), "to", max(counts))
print("Most common tokens:", counts.most_common(10))

print(
    "Most common token share:",
    f"{most_common_count / total_tokens:.2%}",
    f"(ID {most_common_token})"
)

for record in records[:3]:
    print(
        "\nID:", record["utterance_id"],
        "\nText:", record["answer"],
        "\nDuration:", round(record["duration_seconds"], 2),
        "\nToken count:", record["num_speech_tokens"],
        "\nFirst 20:", record["speech_tokens"][:20]
    )

print("\nJSON saved:", OUTPUT_JSON.resolve())
print("Token mapping saved:", TOKEN_PT.resolve())
print("Skipped-sample log:", ERROR_JSON.resolve())