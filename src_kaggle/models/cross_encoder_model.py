"""Binary semantic pair model with transformer-ready interface.

Backend `transformers` is available for real cross-encoder fine-tuning. Backend
`sklearn_text` is a deterministic pair-text classifier for CPU-safe OOF runs and
ablation; it is not demo fallback logic, it is an explicit configurable backend.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.models.pair_text_builder import add_pair_text


class CrossEncoderModel:
    def __init__(self, backend: str = "sklearn_text", model_name: str = "dbmdz/distilbert-base-turkish-cased", text_format_version: str = "full_v1", seed: int = 42, params: dict | None = None):
        self.backend = backend
        self.model_name = model_name
        self.text_format_version = text_format_version
        self.seed = seed
        self.params = params or {}
        self.pipeline = None

    def _texts(self, df: pd.DataFrame) -> list[str]:
        return add_pair_text(df, self.text_format_version)["pair_text"].fillna("").astype(str).tolist()

    def fit(self, df: pd.DataFrame, y) -> "CrossEncoderModel":
        texts = self._texts(df)
        if self.backend == "transformers":
            from src_kaggle.models.transformer_trainer import TransformerCrossEncoder
            self.transformer = TransformerCrossEncoder({
                **self.params,
                "model_name": self.model_name,
                "tokenizer_name": self.params.get("tokenizer_name"),
                "text_format_version": self.text_format_version,
                "max_length": self.params.get("max_length", 256),
                "seed": self.seed,
            })
            raise RuntimeError("Use TransformerCrossEncoder.fit(train_df, val_df, output_dir) via training/train_cross_encoder.py for backend=transformers; no silent sklearn fallback is allowed.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=self.params.get("max_features", 250_000), ngram_range=tuple(self.params.get("ngram_range", (1, 2))), min_df=1)),
            ("clf", LogisticRegression(C=self.params.get("C", 4.0), max_iter=self.params.get("max_iter", 1000), class_weight="balanced", random_state=self.seed, n_jobs=-1)),
        ])
        self.pipeline.fit(texts, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        return self.pipeline.predict_proba(self._texts(df))[:, 1]

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "CrossEncoderModel":
        return joblib.load(path)
