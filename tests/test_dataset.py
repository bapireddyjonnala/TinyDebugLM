import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.dataset import DebugDataset

dataset = DebugDataset(
    "data/debug_corpus.json"
)

print("Dataset size:", len(dataset))

print()

print(dataset[0][:500])