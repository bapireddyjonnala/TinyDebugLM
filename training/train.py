from training.config import TrainingConfig
from training.dataset import DebugDataset
from training.trainer import Trainer

from models.transformer import TinyDebugLM


def main():

    config = TrainingConfig()

    dataset = DebugDataset(
        json_file="data/debug_corpus.json",
        tokenizer_path="tokenizer/tokenizer_output/tokenizer.json",
        max_length=config.max_length
    )

    model = TinyDebugLM(
        vocab_size=config.vocab_size,
        max_length=config.max_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )

    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config
    )

    trainer.train()


if __name__ == "__main__":
    main()