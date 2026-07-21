import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        dataset,
        config
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        self.loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        self.criterion = nn.CrossEntropyLoss()

        self.config = config

        os.makedirs("checkpoints", exist_ok=True)

    def train(self):

        self.model.train()

        for epoch in range(self.config.epochs):

            total_loss = 0

            progress = tqdm(self.loader)

            for batch in progress:

                batch = batch.to(self.device)

                inputs = batch[:, :-1]

                targets = batch[:, 1:]

                logits = self.model(inputs)

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

                progress.set_description(
                    f"Epoch {epoch+1}"
                )

                progress.set_postfix(
                    loss=loss.item()
                )

            avg_loss = total_loss / len(self.loader)

            print(
                f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}"
            )

            torch.save(
                self.model.state_dict(),
                self.config.save_path
            )

        print("Training Finished.")