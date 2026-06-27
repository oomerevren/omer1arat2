"""Transformer tokenization/input prep for cross-encoder fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.models.pair_text_builder import add_pair_text


@dataclass
class TokenLengthStats:
    n_rows: int
    avg_token_length: float
    p95_token_length: float
    max_token_length: int
    truncation_rate: float
    max_length: int
    text_format_version: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_pair_texts(df: pd.DataFrame, text_format_version: str) -> list[str]:
    return add_pair_text(df, text_format_version)["pair_text"].fillna("").astype(str).tolist()


def token_length_diagnostics(df: pd.DataFrame, tokenizer, text_format_version: str, max_length: int) -> TokenLengthStats:
    texts = build_pair_texts(df, text_format_version)
    lengths = []
    for t in texts:
        encoded = tokenizer(t, add_special_tokens=True, truncation=False, padding=False)
        lengths.append(len(encoded["input_ids"]))
    arr = np.asarray(lengths, dtype=int) if lengths else np.asarray([0])
    return TokenLengthStats(
        n_rows=len(texts),
        avg_token_length=float(arr.mean()),
        p95_token_length=float(np.percentile(arr, 95)),
        max_token_length=int(arr.max()),
        truncation_rate=float((arr > max_length).mean()) if len(arr) else 0.0,
        max_length=max_length,
        text_format_version=text_format_version,
    )


class PairClassificationDataset:
    def __init__(self, df: pd.DataFrame, tokenizer, text_format_version: str = "full_v1", max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.texts = build_pair_texts(self.df, text_format_version)
        self.labels = self.df[SCHEMA.label].astype(int).tolist() if SCHEMA.label in self.df.columns else None
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item
