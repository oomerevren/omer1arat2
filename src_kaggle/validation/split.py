"""Validation utilities for pair classification."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src_kaggle.data.schema import SCHEMA


def make_stratified_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(out, out[SCHEMA.label].astype(int))):
        out.loc[out.index[valid_idx], "fold"] = fold
    return out
