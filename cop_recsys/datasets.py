from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SessionDataset(Dataset):
    def __init__(self, data_dict: dict[str, torch.Tensor | None], num_items: int, is_train: bool = False):
        self.data = data_dict
        self.num_items = num_items
        self.is_train = is_train
        self.length = len(self.data["target"])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.data.items() if val is not None}
        if self.is_train:
            target = item["target"].item()
            while True:
                neg = np.random.randint(1, self.num_items)
                if neg != target:
                    break
            item["neg_item"] = torch.tensor(neg, dtype=torch.long)
        return item


def create_session_dataloaders(
    train_data: dict[str, torch.Tensor | None],
    valid_data: dict[str, torch.Tensor | None],
    test_data: dict[str, torch.Tensor | None],
    num_items: int,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = SessionDataset(train_data, num_items, is_train=True)
    valid_dataset = SessionDataset(valid_data, num_items, is_train=False)
    test_dataset = SessionDataset(test_data, num_items, is_train=False)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
