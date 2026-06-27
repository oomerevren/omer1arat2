"""Standard OOF prediction schema and metric helpers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src_kaggle.data.schema import SCHEMA


@dataclass
class OOFResult:
    model_name: str
    oof_path: str
    artifact_dir: str
    best_threshold: float
    macro_f1: float
    fold_scores: list[float]
    report_path: str | None = None


def find_best_threshold(y_true, proba, grid=None) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 91)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = f1_score(y_true, (proba >= t).astype(int), average="macro")
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42, group_col: str = SCHEMA.term_id) -> pd.Series:
    y = df[SCHEMA.label].astype(int).values
    folds = pd.Series(-1, index=df.index, dtype=int)
    if group_col in df.columns and df[group_col].nunique() >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        for fold, (_, va) in enumerate(splitter.split(df, y, groups=df[group_col])):
            folds.iloc[va] = fold
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, va) in enumerate(splitter.split(df, y)):
            folds.iloc[va] = fold
    return folds


def build_oof_frame(df: pd.DataFrame, proba, model_name: str, fold_col="fold") -> pd.DataFrame:
    cols = [c for c in [SCHEMA.id, SCHEMA.term_id, SCHEMA.item_id, SCHEMA.label, fold_col] if c in df.columns]
    out = df[cols].copy()
    out["model_name"] = model_name
    out["proba"] = np.asarray(proba, dtype=float)
    return out


def write_json(data, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
