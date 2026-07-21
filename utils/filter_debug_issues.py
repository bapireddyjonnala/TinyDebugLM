import json
from pathlib import Path

INPUT_FILE = Path("data/final/instruction_dataset.json")
OUTPUT_FILE = Path("data/final/debug_only_dataset.json")

DEBUG_KEYWORDS = [
    "bug",
    "error",
    "exception",
    "crash",
    "failure",
    "fails",
    "issue",
    "traceback",
    "module",
    "import",
    "cannot",
    "segmentation",
    "memory",
    "timeout",
    "install",
    "dependency",
]


def is_debug_issue(example):

    title = example["input"]["title"].lower()
    description = example["input"]["description"].lower()

    labels = " ".join(example["input"]["labels"]).lower()

    text = f"{title} {description} {labels}"

    return any(keyword in text for keyword in DEBUG_KEYWORDS)


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    filtered = [
        item for item in dataset
        if is_debug_issue(item)
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print("Debug Examples :", len(filtered))
    print("Removed        :", len(dataset) - len(filtered))
    print("Saved to       :", OUTPUT_FILE)
    print("=" * 50)


if __name__ == "__main__":
    main()