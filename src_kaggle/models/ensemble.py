"""OOF-based blending utilities."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src_kaggle.models.oof import find_best_threshold
from src_kaggle.data.schema import SCHEMA


class WeightedAverageEnsemble:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {}
        self.best_threshold = 0.5

    def fit_oof(self, oof_frames: dict[str, pd.DataFrame]) -> dict:
        names = list(oof_frames)
        if not self.weights:
            self.weights = {n: 1/len(names) for n in names}
        base = oof_frames[names[0]][[SCHEMA.id, SCHEMA.label]].copy()
        proba = np.zeros(len(base), dtype=float)
        for n in names:
            proba += float(self.weights.get(n, 0)) * oof_frames[n]["proba"].values
        self.best_threshold, score = find_best_threshold(base[SCHEMA.label].values, proba)
        return {"models": names, "weights": self.weights, "best_threshold": self.best_threshold, "macro_f1": score}

    def predict(self, pred_frames: dict[str, pd.DataFrame]) -> np.ndarray:
        names = list(pred_frames)
        proba = np.zeros(len(pred_frames[names[0]]), dtype=float)
        for n in names:
            proba += float(self.weights.get(n, 0)) * pred_frames[n]["proba"].values
        return proba

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path):
        return joblib.load(path)
