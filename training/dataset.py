import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class DebugDataset(Dataset):

    def __init__(
        self,
        json_file,
        tokenizer_path,
        max_length=256
    ):

        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        text = (
            sample["instruction"]
            + "\n"
            + sample["input"]["title"]
            + "\n"
            + sample["input"]["description"]
        )

        encoding = self.tokenizer.encode(text)

        ids = encoding.ids[:self.max_length]

        if len(ids) < self.max_length:
            ids += [0] * (self.max_length - len(ids))

        return torch.tensor(ids)