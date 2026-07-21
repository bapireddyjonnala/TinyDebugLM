import json
from pathlib import Path

INPUT_DIR = Path("data/raw/github_issues")
OUTPUT_DIR = Path("data/processed/github_issues")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = list(INPUT_DIR.glob("*.json"))

print(f"Found {len(files)} repositories")


for file in files:

    print(f"Processing {file.name}")

    with open(file, "r", encoding="utf-8") as f:
        issues = json.load(f)

    cleaned = []

    for issue in issues:

        # Skip Pull Requests
        if "pull_request" in issue:
            continue

        # Skip empty issues
        if not issue.get("body"):
            continue

        cleaned.append({

            "repository": file.stem,

            "title": issue.get("title"),

            "body": issue.get("body"),

            "labels": [
                label["name"]
                for label in issue.get("labels", [])
            ],

            "state": issue.get("state"),

            "comments": issue.get("comments")

        })

    output = OUTPUT_DIR / file.name

    with open(output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=4)

    print(f"Saved {len(cleaned)} issues")

print("\nDone.")