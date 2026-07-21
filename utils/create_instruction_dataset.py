import json
from pathlib import Path

INPUT_FILE = Path("data/final/debug_dataset.json")
OUTPUT_FILE = Path("data/final/instruction_dataset.json")


def convert_issue(issue):

    title = issue.get("title", "").strip()

    description = (
        issue.get("body", "") or
        issue.get("description", "")
    )

    labels = []

    for label in issue.get("labels", []):

        if isinstance(label, dict):
            labels.append(label.get("name", ""))
        else:
            labels.append(str(label))

    return {
        "instruction": "Analyze this software issue and explain the possible cause and solution.",

        "input": {
            "title": title,
            "description": description,
            "labels": labels
        },

        "output": {
            "cause": "",
            "solution": "",
            "explanation": ""
        },

        "metadata": {
            "source": "GitHub",
            "url": issue.get("html_url", ""),
            "state": issue.get("state", "")
        }
    }


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = []

    for issue in raw:

        if issue.get("title"):

            dataset.append(
                convert_issue(issue)
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 50)
    print("Instruction Dataset Created")
    print("Examples:", len(dataset))
    print("Saved to:", OUTPUT_FILE)
    print("=" * 50)


if __name__ == "__main__":
    main()