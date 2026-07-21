import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from models.loss import LanguageModelLoss

loss_fn = LanguageModelLoss()

logits = torch.randn(
    2,
    20,
    16000
)

targets = torch.randint(
    0,
    16000,
    (2, 20)
)

loss = loss_fn(
    logits,
    targets
)

print(loss)