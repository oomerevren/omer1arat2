"""Lightweight ensemble wrapper: CrossEncoder + tabular feature model fusion."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data.dataset import get_num_labels
from src.features.feature_pipeline import FeaturePipeline
from src.models.cross_encoder import CrossEncoderModel
from src.utils.io import read_json, write_json


class EnsembleModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.num_labels = get_num_labels(self.config) if self.config else 3
        ce_cfg = self.config.get("model", {}).get("cross_encoder", {})
        self.cross_encoder = CrossEncoderModel(
            model_name=ce_cfg.get("model_name", "dbmdz/distilbert-base-turkish-cased"),
            num_labels=self.num_labels,
            max_length=ce_cfg.get("max_length", 256),
            load_pretrained=False,
        )
        self.feature_pipeline = FeaturePipeline(self.config)
        self.models = []
        self.weights = np.array([0.70, 0.30], dtype=np.float32)

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        config = config or self.config
        self.config = config
        self.num_labels = get_num_labels(config)
        metrics = self.cross_encoder.train(train_df, config)

        label_col = config.get("data", {}).get("label_column", "is_relevant")
        y = train_df[label_col].astype(int).clip(0, self.num_labels - 1).values
        X, _ = self.feature_pipeline.fit_transform(train_df)

        self.models = []
        if len(np.unique(y)) < 2:
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(X, y)
            self.models.append(clf)
        else:
            candidates = [
                make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=500, class_weight="balanced")),
                RandomForestClassifier(n_estimators=80, random_state=42, class_weight="balanced_subsample"),
            ]
            # GradientBoosting küçük sample'da stabil olmayabilir; denenir, hata olursa atlanır.
            candidates.append(GradientBoostingClassifier(random_state=42))
            for clf in candidates:
                try:
                    clf.fit(X, y)
                    self.models.append(clf)
                except Exception as exc:
                    print(f"[EnsembleModel] Tabular model atlandı: {exc}")

        preds = self.predict(train_df, config)
        metrics.update({
            "ensemble_train_macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
            "ensemble_model_count": float(len(self.models) + 1),
        })
        return metrics

    def _align_proba(self, raw: np.ndarray, classes) -> np.ndarray:
        probs = np.zeros((raw.shape[0], self.num_labels), dtype=np.float32)
        for j, cls in enumerate(list(classes)):
            if int(cls) < self.num_labels:
                probs[:, int(cls)] = raw[:, j]
        row_sum = probs.sum(axis=1, keepdims=True)
        probs = probs / np.clip(row_sum, 1e-8, None)
        return probs

    def predict_proba(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        config = config or self.config
        ce_probs = self.cross_encoder.predict_proba(df, config)
        if not self.models:
            return ce_probs
        X = self.feature_pipeline.transform(df)
        table_probs = []
        for clf in self.models:
            raw = clf.predict_proba(X)
            classes = getattr(clf, "classes_", None)
            if classes is None and hasattr(clf, "steps"):
                classes = clf.steps[-1][1].classes_
            table_probs.append(self._align_proba(raw, classes))
        avg_table = np.mean(table_probs, axis=0)
        w_ce = float(self.weights[0])
        w_tab = float(self.weights[1])
        out = w_ce * ce_probs + w_tab * avg_table
        return out / np.clip(out.sum(axis=1, keepdims=True), 1e-8, None)

    def predict(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return np.argmax(self.predict_proba(df, config), axis=1)

    def evaluate(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        config = config or self.config
        label_col = config.get("data", {}).get("label_column", "is_relevant")
        if label_col not in df.columns or len(df) == 0:
            return {"macro_f1": 0.0, "precision": 0.0, "recall": 0.0}
        y_true = df[label_col].astype(int).clip(0, self.num_labels - 1).values
        y_pred = self.predict(df, config)
        return {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.cross_encoder.save(path / "cross_encoder")
        with open(path / "ensemble.pkl", "wb") as f:
            pickle.dump({"models": self.models, "feature_pipeline": self.feature_pipeline, "weights": self.weights}, f)
        write_json(path / "meta.json", {"num_labels": self.num_labels, "model_type": "ensemble"})

    @classmethod
    def load(cls, path: str | Path, config: Optional[Dict[str, Any]] = None) -> "EnsembleModel":
        path = Path(path)
        obj = cls(config)
        ce_path = path / "cross_encoder"
        if ce_path.exists():
            obj.cross_encoder = CrossEncoderModel.load(ce_path)
        pkl = path / "ensemble.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            obj.models = data.get("models", [])
            obj.feature_pipeline = data.get("feature_pipeline", obj.feature_pipeline)
            obj.weights = data.get("weights", obj.weights)
        meta_path = path / "meta.json"
        if meta_path.exists():
            obj.num_labels = int(read_json(meta_path).get("num_labels", obj.num_labels))
        return obj
