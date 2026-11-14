from huggingface_hub import snapshot_download

REPO_ID = "nisha-speech-ai/mandarin-english-cs-asr-tts"

print("Downloading all models and checkpoints from Hugging Face...")
print("Please wait, this may take a few minutes depending on internet speed...")

# This automatically downloads all folders (ascend_lora_training, whisper_runs,
# whisper_models, pretrained_models) into your current project folder
snapshot_download(
    repo_id=REPO_ID,
    repo_type="model",
    local_dir=".",
    local_dir_use_symlinks=False
)

print("All models downloaded and placed in their exact folders!")
print("You can now run: python asr_tts_studio.py")
