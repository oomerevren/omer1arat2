"""Index build/load helpers for hybrid retrieval."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src_kaggle.retrieval.hybrid_retriever import HybridRetriever


def build_retrieval_index(items: pd.DataFrame, cfg: dict) -> HybridRetriever:
    return HybridRetriever.build(items, seed=int(cfg.get("seed", 42)), retrieval_cfg=cfg)


def save_retrieval_index(index: HybridRetriever, path: str | Path) -> None:
    index.save(path)


def load_retrieval_index(path: str | Path) -> HybridRetriever:
    return HybridRetriever.load(path)
