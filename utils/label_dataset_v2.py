import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# Load Environment
# -------------------------
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# -------------------------
# Configuration
# -------------------------
MODEL = "openai/gpt-oss-20b:free"

INPUT_FILE = "data/final/debug_only_dataset.json"
OUTPUT_FILE = "data/final/debug_labeled_dataset.json"

# -------------------------
# Load previous progress
# -------------------------
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        labeled = json.load(f)
else:
    labeled = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

completed = len(labeled)

print("=" * 60)
print(f"Already labeled : {completed}")
print(f"Remaining       : {len(dataset) - completed}")
print("=" * 60)

dataset = dataset[completed:]

# ---------------------------------------------------
# Test with ONE example
# Change dataset[:1] -> dataset when it works
# ---------------------------------------------------
for i, example in enumerate(dataset, start=completed + 1):

    title = example["input"]["title"]
    description = example["input"]["description"]

    prompt = f"""
You are an expert software engineer.

Analyze the following GitHub issue.

Title:
{title}

Description:
{description}

Return ONLY valid JSON.

{{
  "cause": "...",
  "solution": "...",
  "explanation": "..."
}}
"""

    success = False

    for attempt in range(5):

        try:

            print(f"\nExample {i} | Attempt {attempt + 1}")

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2
            )

            text = response.choices[0].message.content.strip()

            # Remove markdown fences
            if text.startswith("```json"):
                text = text.replace("```json", "", 1)

            if text.startswith("```"):
                text = text.replace("```", "", 1)

            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

            print("\nModel Response:\n")
            print(text)

            start = text.find("{")
            end = text.rfind("}") + 1

            labels = json.loads(text[start:end])

            example["output"] = labels

            labeled.append(example)

            with open(
                OUTPUT_FILE,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    labeled,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            print(f"\n✓ Example {i} saved successfully!")

            success = True
            break

        except Exception as e:

            print("\nError:")
            print(e)

            if "429" in str(e):
              print("Rate limited. Waiting 20 seconds...")
              time.sleep(20)
              continue

            break

    if not success:
        print(f"\n✗ Failed to label example {i}")

print("\nFinished.")