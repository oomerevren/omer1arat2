"""Etiket normalizasyonu — Kaggle (binary) ve Final (3-sınıf) modları."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.data.dataset import get_num_labels


def normalize_labels(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Moda göre etiketleri standartlaştırır."""
    out = df.copy()
    label_col = config.get("data", {}).get("label_column", "is_relevant")
    if label_col not in out.columns:
        return out

    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)
    num_labels = get_num_labels(config)

    if num_labels == 2:
        # Kaggle: pozitif (1,2,...) -> 1, negatif -> 0
        out[label_col] = (out[label_col] >= 1).astype(int)
    elif num_labels == 3:
        out[label_col] = out[label_col].clip(0, 2)

    return out


def is_kaggle_mode(config: Dict[str, Any]) -> bool:
    return get_num_labels(config) == 2
