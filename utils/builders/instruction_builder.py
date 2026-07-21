import json
from pathlib import Path

INPUT_DIR = Path("data/processed/github_issues")
OUTPUT_DIR = Path("data/instruction")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

instruction_dataset = []

for file in INPUT_DIR.glob("*.json"):

    print(f"Processing {file.name}")

    with open(file, "r", encoding="utf-8") as f:
        issues = json.load(f)

    for issue in issues:

        sample = {
            "instruction": "Analyze this software issue and explain the possible cause and solution.",
            "input": {
                "repository": issue["repository"],
                "title": issue["title"],
                "description": issue["body"],
                "labels": issue["labels"],
                "state": issue["state"]
            },
            "output": {
                "cause": "",
                "solution": "",
                "explanation": ""
            },
            "metadata": {
                "source": "GitHub",
                "comments": issue["comments"]
            }
        }

        instruction_dataset.append(sample)

output_file = OUTPUT_DIR / "github_instruction_dataset.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(instruction_dataset, f, indent=4)

print(f"\nCreated {len(instruction_dataset)} instruction samples.")
print(f"Saved to {output_file}")