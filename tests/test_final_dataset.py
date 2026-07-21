import json

with open("data/final/debug_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total Examples:", len(data))

print("\nFirst Example:\n")
print(json.dumps(data[0], indent=2)[:2000])