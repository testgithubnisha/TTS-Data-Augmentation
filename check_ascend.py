from datasets import load_dataset, Audio

dataset = load_dataset("CAiRE/ASCEND")

# Inspect records without triggering audio decoding.
for split in dataset:
    dataset[split] = dataset[split].cast_column(
        "audio",
        Audio(decode=False)
    )

print(dataset)

sample = dataset["train"][0]

print("\nColumns:", dataset["train"].column_names)
print("Transcript:", sample["transcription"])
print("Duration:", sample["duration"])
print("Speaker ID:", sample["original_speaker_id"])
print("Language:", sample["language"])

