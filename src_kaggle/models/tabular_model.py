"""Tabular feature-based binary models for Kaggle War Mode."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class TabularModel:
    def __init__(self, model_type: str = "hist_gradient_boosting", params: dict | None = None, seed: int = 42):
        self.model_type = model_type
        self.params = params or {}
        self.seed = seed
        self.model = self._make_model()
        self.feature_names_: list[str] = []

    def _make_model(self):
        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(random_state=self.seed, **self.params)
            except Exception:
                self.model_type = "hist_gradient_boosting"
        if self.model_type == "catboost":
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(random_seed=self.seed, verbose=False, **self.params)
            except Exception:
                self.model_type = "hist_gradient_boosting"
        from sklearn.ensemble import HistGradientBoostingClassifier
        defaults = {"learning_rate": 0.06, "max_iter": 300, "l2_regularization": 0.01, "random_state": self.seed}
        defaults.update(self.params)
        return HistGradientBoostingClassifier(**defaults)

    def fit(self, X: pd.DataFrame, y) -> "TabularModel":
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.feature_names_, fill_value=0) if self.feature_names_ else X
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        raw = self.model.decision_function(X)
        return 1 / (1 + np.exp(-raw))

    def feature_importance(self) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            vals = self.model.feature_importances_
        else:
            vals = np.zeros(len(self.feature_names_))
        return pd.DataFrame({"feature": self.feature_names_, "importance": vals}).sort_values("importance", ascending=False)

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TabularModel":
        return joblib.load(path)
