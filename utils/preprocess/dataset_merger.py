import json
from pathlib import Path

INPUT_DIR = Path("data/instruction")
OUTPUT_FILE = Path("data/debug_corpus.json")

merged = []

for file in INPUT_DIR.glob("*.json"):
    print(f"Reading {file.name}")

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        merged.extend(data)
    else:
        merged.append(data)

print(f"\nTotal samples: {len(merged)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=4)

print(f"Saved to {OUTPUT_FILE}")