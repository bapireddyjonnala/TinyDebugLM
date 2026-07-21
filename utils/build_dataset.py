import json
from pathlib import Path

RAW_DIR = Path("data/raw/github_issues")
OUTPUT = Path("data/final/debug_dataset.json")

all_examples = []

for file in RAW_DIR.glob("*.json"):
    with open(file, "r", encoding="utf-8") as f:
        issues = json.load(f)

    for issue in issues:
        all_examples.append(issue)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(all_examples, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_examples)} examples to {OUTPUT}")