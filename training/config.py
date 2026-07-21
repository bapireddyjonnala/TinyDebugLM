from dataclasses import dataclass


@dataclass
class TrainingConfig:

    vocab_size = 16000

    max_length = 256

    embedding_dim = 256

    hidden_dim = 1024

    num_heads = 8

    num_layers = 6

    dropout = 0.1

    batch_size = 4

    epochs = 5

    learning_rate = 3e-4

    save_path = "checkpoints/tinydebuglm.pt"