import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.dataset import DebugDataset

dataset = DebugDataset(
    json_file="data/debug_corpus.json",
    tokenizer_path="tokenizer/tokenizer_output/tokenizer.json",
    max_length=64
)

sample = dataset[0]

print(sample)
print()
print(sample.shape)