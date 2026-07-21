import os
import json
import requests
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
GITHUB_API = "https://api.github.com"

# Public repositories to collect issues from
REPOSITORIES = [
    "pytorch/pytorch",
    "python/cpython",
    "numpy/numpy",
    "pandas-dev/pandas",
    "fastapi/fastapi",
    "huggingface/transformers"
]

# Number of issues to download per repository
ISSUES_PER_REPO = 100

# Output directory
OUTPUT_DIR = Path("data/raw/github_issues")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional GitHub Personal Access Token
TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Accept": "application/vnd.github+json"
}

if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"


def fetch_issues(repo_name):
    """Download issues from a GitHub repository."""

    print(f"\nDownloading issues from {repo_name}")

    url = f"{GITHUB_API}/repos/{repo_name}/issues"

    params = {
        "state": "all",
        "per_page": ISSUES_PER_REPO
    }

    response = requests.get(url, headers=HEADERS, params=params)

    response.raise_for_status()

    return response.json()


def save_issues(repo_name, issues):
    """Save issues as JSON."""

    filename = repo_name.replace("/", "_") + ".json"

    filepath = OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=4)

    print(f"Saved -> {filepath}")


def main():

    print("=" * 50)
    print("TinyDebugLM GitHub Dataset Collector")
    print("=" * 50)

    for repo in REPOSITORIES:
        try:
            issues = fetch_issues(repo)
            save_issues(repo, issues)

        except Exception as e:
            print(f"Failed: {repo}")
            print(e)


if __name__ == "__main__":
    main()