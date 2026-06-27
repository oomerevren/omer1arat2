"""Fast baseline model for Kaggle War Mode."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.text_features import PairTextVectorizer


class TfidfLogRegPairClassifier:
    def __init__(self, C: float = 4.0, max_iter: int = 1000, seed: int = 42) -> None:
        self.vectorizer = PairTextVectorizer()
        self.model = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", random_state=seed, n_jobs=-1)

    def fit(self, df: pd.DataFrame) -> "TfidfLogRegPairClassifier":
        X = self.vectorizer.fit_transform(df)
        y = df[SCHEMA.label].astype(int).values
        self.model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.vectorizer.transform(df)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TfidfLogRegPairClassifier":
        return joblib.load(path)
