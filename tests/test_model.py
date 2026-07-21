import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from models.transformer import TinyDebugLM

model = TinyDebugLM(
    vocab_size=16000,
    max_length=512
)

x = torch.randint(
    0,
    16000,
    (2, 20)
)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)