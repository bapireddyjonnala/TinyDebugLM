import torch
from tokenizers import Tokenizer

from models.transformer import TinyDebugLM
from training.config import TrainingConfig


class TinyDebugGenerator:

    def __init__(self):

        self.config = TrainingConfig()

        self.tokenizer = Tokenizer.from_file(
            "tokenizer/tokenizer_output/tokenizer.json"
        )

        self.model = TinyDebugLM(
            vocab_size=self.config.vocab_size,
            max_length=self.config.max_length,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        self.model.load_state_dict(
            torch.load(
                self.config.save_path,
                map_location="cpu"
            )
        )

        self.model.eval()

    @torch.no_grad()
    def predict_next_token(self, text):

        encoding = self.tokenizer.encode(text)

        ids = encoding.ids

        if len(ids) > self.config.max_length:
            ids = ids[:self.config.max_length]

        x = torch.tensor([ids])

        logits = self.model(x)

        next_token = torch.argmax(
            logits[0, -1]
        ).item()

        return next_token