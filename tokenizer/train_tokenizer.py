import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

DATASET = Path("data/debug_corpus.json")

texts = []

with open(DATASET, "r", encoding="utf-8") as f:
    dataset = json.load(f)

for sample in dataset:

    texts.append(sample["instruction"])

    texts.append(sample["input"]["title"])

    texts.append(sample["input"]["description"])

    if sample["output"]["cause"]:
        texts.append(sample["output"]["cause"])

    if sample["output"]["solution"]:
        texts.append(sample["output"]["solution"])

    if sample["output"]["explanation"]:
        texts.append(sample["output"]["explanation"])

corpus = Path("tokenizer/corpus.txt")

with open(corpus, "w", encoding="utf-8") as f:
    f.write("\n".join(texts))

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=16000,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]"
    ]
)

tokenizer.train(
    [str(corpus)],
    trainer
)

output_dir = Path("tokenizer/tokenizer_output")
output_dir.mkdir(exist_ok=True)

tokenizer.save(str(output_dir / "tokenizer.json"))

print("Tokenizer trained successfully!")