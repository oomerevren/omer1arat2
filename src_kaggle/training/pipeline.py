"""Single canonical Kaggle War Mode training/validation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src_kaggle.data.schema import SCHEMA
from src_kaggle.models.baseline import TfidfLogRegPairClassifier
from src_kaggle.validation.split import make_stratified_folds


@dataclass
class ValidationResult:
    macro_f1: float
    threshold: float
    fold_scores: list[float]


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = f1_score(y_true, (proba >= threshold).astype(int), average="macro")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def cross_validate_baseline(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> ValidationResult:
    folded = make_stratified_folds(df, n_splits=n_splits, seed=seed)
    oof = np.zeros(len(folded), dtype=float)
    fold_scores: list[float] = []
    for fold in range(n_splits):
        train_df = folded[folded["fold"] != fold].reset_index(drop=True)
        valid_df = folded[folded["fold"] == fold].reset_index(drop=True)
        model = TfidfLogRegPairClassifier(seed=seed + fold)
        model.fit(train_df)
        proba = model.predict_proba(valid_df)
        oof[folded["fold"].values == fold] = proba
        threshold, score = find_best_threshold(valid_df[SCHEMA.label].astype(int).values, proba)
        fold_scores.append(score)
        print(f"fold={fold} macro_f1={score:.6f} threshold={threshold:.3f}")
    threshold, macro_f1 = find_best_threshold(folded[SCHEMA.label].astype(int).values, oof)
    return ValidationResult(macro_f1=macro_f1, threshold=threshold, fold_scores=fold_scores)


def train_full_baseline(df: pd.DataFrame, seed: int = 42) -> TfidfLogRegPairClassifier:
    model = TfidfLogRegPairClassifier(seed=seed)
    model.fit(df)
    return model
