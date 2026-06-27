"""Lightweight BM25 retriever with persist/reload support."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.retrieval.item_text_builder import tokenize


@dataclass
class BM25Retriever:
    items: pd.DataFrame
    tokenized_corpus: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self):
        self.doc_len = np.array([len(d) for d in self.tokenized_corpus], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.term_freqs = [Counter(doc) for doc in self.tokenized_corpus]
        df = Counter()
        for doc in self.tokenized_corpus:
            df.update(set(doc))
        n_docs = max(1, len(self.tokenized_corpus))
        self.idf = {t: math.log(1 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def fit(cls, items: pd.DataFrame, text_col: str = SCHEMA.retrieval_text) -> "BM25Retriever":
        return cls(items=items.reset_index(drop=True), tokenized_corpus=[tokenize(t) for t in items[text_col].fillna("").astype(str)])

    def scores(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        scores = np.zeros(len(self.items), dtype=float)
        if not q_terms or not len(self.items):
            return scores
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1e-9))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores

    def search(self, query: str, top_k: int = 50, exclude_items: set | None = None) -> pd.DataFrame:
        exclude_items = exclude_items or set()
        scores = self.scores(query)
        order = np.argsort(-scores)
        rows = []
        for idx in order:
            row = self.items.iloc[idx].copy()
            if row[SCHEMA.item_id] in exclude_items:
                continue
            if scores[idx] <= 0 and rows:
                break
            row["source"] = "bm25"
            row["score"] = float(scores[idx])
            row["bm25_score"] = float(scores[idx])
            rows.append(row)
            if len(rows) >= top_k:
                break
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "BM25Retriever":
        with Path(path).open("rb") as f:
            return pickle.load(f)
