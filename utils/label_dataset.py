import json
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

INPUT_FILE = "data/final/debug_only_dataset.json"
OUTPUT_FILE = "data/final/debug_labeled_dataset.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []

for i, example in enumerate(dataset):

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
  "cause":"...",
  "solution":"...",
  "explanation":"..."
}}
"""

    try:

        response = model.generate_content(prompt)

        text = response.text.strip()

        start = text.find("{")
        end = text.rfind("}") + 1

        labels = json.loads(text[start:end])

        example["output"] = labels

        results.append(example)

        print(f"[{i+1}/{len(dataset)}] OK")

    except Exception as e:

        print(f"[{i+1}] Failed:", e)

    time.sleep(2)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Finished.")