# src/dataset.py
import json
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import format_input, normalize_text
from config import SETTINGS


class HallucinationDataset(Dataset):
    
    def __init__(
        self,
        rows: List[dict],
        tokenizer,
        max_length: int = 256,
    ):
        self.rows      = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        answer    = normalize_text(str(row.get("text",      "")))
        question  = normalize_text(str(row.get("question",  "")))
        knowledge = normalize_text(str(row.get("knowledge", "")))
        label     = int(row.get("label", 0))

        text = format_input(answer, question, knowledge)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def load_json_dataset(path: str) -> List[dict]:
    """Load combined_dataset.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_dataset(
    data: List[dict],
    train_size: float = 0.60,
    val_size:   float = 0.20,
    random_seed: int  = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    labels = [r["label"] for r in data]

    train_val, test = train_test_split(
        data, test_size=0.20, stratify=labels, random_state=random_seed
    )
    tv_labels = [r["label"] for r in train_val]
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=tv_labels, random_state=random_seed
    )
    return train, val, test


def make_dataloader(
    rows: List[dict],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a list of dataset rows."""
    dataset = HallucinationDataset(rows, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
