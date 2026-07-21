import torch
import torch.nn as nn


class LanguageModelLoss(nn.Module):
    """
    Cross Entropy Loss for Next Token Prediction
    """

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):

        vocab_size = logits.size(-1)

        logits = logits.view(-1, vocab_size)

        targets = targets.reshape(-1)

        loss = self.loss_fn(
            logits,
            targets
        )

        return loss