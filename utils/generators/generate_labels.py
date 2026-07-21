import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INPUT_FILE = Path("data/instruction/github_instruction_dataset.json")
OUTPUT_FILE = Path("data/instruction/github_instruction_dataset_labeled.json")

# -----------------------------
# Load input dataset
# -----------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# -----------------------------
# Resume support
# -----------------------------
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        completed = json.load(f)
else:
    completed = []

completed_titles = {
    sample["input"]["title"]
    for sample in completed
}

print(f"Already completed: {len(completed)}")

# -----------------------------
# Generate labels
# -----------------------------
for sample in dataset:

    if sample["input"]["title"] in completed_titles:
        continue

    print(f"\nProcessing: {sample['input']['title']}")

    prompt = f"""
You are an expert software engineer.

Analyze this GitHub issue.

Repository:
{sample['input']['repository']}

Title:
{sample['input']['title']}

Description:
{sample['input']['description']}

Return ONLY JSON.

{{
"cause":"",
"solution":"",
"explanation":""
}}
"""

    while True:

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            text = (
                response.text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            answer = json.loads(text)

            sample["output"] = answer

            completed.append(sample)

            # Save after every sample
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(completed, f, indent=4)

            print("Saved.")

            break

        except Exception as e:

            print(e)

            if "429" in str(e):
                print("Quota exceeded. Waiting 60 seconds...")
                time.sleep(60)
            else:
                print("Skipping...")
                break

print("\nFinished.")