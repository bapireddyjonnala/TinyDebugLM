"""
TinyDomainLM Dataset Downloader

This script downloads high-quality AI/ML datasets from Hugging Face
and saves them into the data/raw directory.
"""

from datasets import load_dataset
from pathlib import Path

# -----------------------------
# Create directories
# -----------------------------
BASE_DIR = Path("data/raw/datasets")
BASE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("TinyDomainLM Dataset Downloader")
print("=" * 50)

# -----------------------------
# Download WikiText-2
# -----------------------------
print("\nDownloading WikiText-2...")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

for split in dataset.keys():
    output_file = BASE_DIR / f"{split}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset[split]:
            text = row["text"].strip()

            if len(text) > 0:
                f.write(text + "\n")

    print(f"Saved: {output_file}")

print("\nDataset download completed!")