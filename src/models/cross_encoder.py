"""
Cross-Encoder model wrapper.

Bu modül iki modda çalışır:
1) Gerçek HuggingFace modeli varsa/yüklenmek istenirse üretim eğitim-inference akışı.
2) Model ağırlığı veya ağır bağımlılıklar yoksa, CI/API/test için deterministik ve hızlı
   sklearn/heuristic fallback. Bu fallback DEMO amaçlıdır; gerçek yarışma metriği iddia etmez.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset import build_product_text
from src.features.feature_pipeline import FeaturePipeline
from src.utils.io import read_json, write_json


@dataclass
class _DummyConfig:
    num_labels: int = 3
    model_name: str = "feature-fallback"


class _DummyModel:
    """HF modelinin `.config.num_labels` yüzeyini taklit eden hafif nesne."""

    def __init__(self, num_labels: int, model_name: str):
        self.config = _DummyConfig(num_labels=num_labels, model_name=model_name)


class CrossEncoderModel:
    LABELS_3 = ["Alakasız", "Kısmen Alakalı", "Çok Alakalı"]
    LABELS_2 = ["Alakasız", "Alakalı"]

    def __init__(
        self,
        model_name: str = "dbmdz/distilbert-base-turkish-cased",
        num_labels: int = 3,
        max_length: int = 256,
        device: Optional[str] = None,
        load_pretrained: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.num_labels = int(num_labels or 3)
        self.max_length = int(max_length or 256)
        self.device = device or os.environ.get("DEVICE", "cpu")
        self._last_inference_ms = 0.0
        self._mode = "feature_fallback"
        self._onnx_session = None
        self._hf_tokenizer = None
        self._feature_pipeline = FeaturePipeline({"model": {"cross_encoder": {"num_labels": self.num_labels}}})
        self._classifier = None

        # Güvenli varsayılan: model indirmeye çalışmaz. Açıkça istenirse HF yüklenir.
        should_load_hf = (
            load_pretrained
            if load_pretrained is not None
            else os.environ.get("DEEP_PIPELINE_LOAD_HF", "0") == "1"
        )
        if should_load_hf:
            self._try_load_hf_model()
        else:
            self.model = _DummyModel(self.num_labels, model_name)

    # ------------------------------------------------------------------
    # Optional HuggingFace loading
    # ------------------------------------------------------------------
    def _try_load_hf_model(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,
            )
            self.model.to(self.device)
            self.model.eval()
            self._torch = torch
            self._mode = "huggingface"
        except Exception as exc:
            print(f"[CrossEncoderModel] HF yüklenemedi, feature fallback aktif: {exc}")
            self.model = _DummyModel(self.num_labels, self.model_name)
            self._mode = "feature_fallback"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _labels(self) -> List[str]:
        return self.LABELS_2 if self.num_labels == 2 else self.LABELS_3

    def _build_inference_df(
        self,
        query: str,
        product: str,
        config: Optional[Dict[str, Any]] = None,
        brand: str = "",
        category: str = "",
        color: str = "",
        material: str = "",
    ) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "search_query": query,
                "product_name": product,
                "brand": brand,
                "category": category,
                "product_color": color,
                "product_material": material,
            }
        ])

    def _ensure_product_columns(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        out = df.copy()
        defaults = {
            "search_query": "",
            "product_name": "",
            "brand": "",
            "category": "",
            "product_color": "",
            "product_material": "",
        }
        for col, default in defaults.items():
            if col not in out.columns:
                out[col] = default
        # Ürün metnini zenginleştirmek için boş product_name varsa build_product_text dene.
        if "product_name" in out.columns:
            empty_mask = out["product_name"].fillna("").astype(str).str.len() == 0
            if empty_mask.any():
                out.loc[empty_mask, "product_name"] = out[empty_mask].apply(
                    lambda r: build_product_text(r, config), axis=1
                )
        return out

    def _heuristic_proba(self, df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
        fp = FeaturePipeline(config or {})
        features = fp.get_feature_dict(df.head(1)) if len(df) == 1 else None
        if len(df) != 1:
            # Çoklu tahminde aynı feature kodunu kullanıp satır satır deterministik skor üret.
            rows = [self._heuristic_proba(df.iloc[[i]].copy(), config)[0] for i in range(len(df))]
            return np.vstack(rows)

        features = features or {}
        brand = min(float(features.get("fuzzy_brand_match_score", 0.0)) / 100.0, 1.0)
        jaccard = min(float(features.get("jaccard_similarity", 0.0)), 1.0)
        overlap = min(float(features.get("text_overlap_ratio", 0.0)), 1.0)
        coverage = min(float(features.get("query_coverage", 0.0)), 1.0)
        bm25 = min(float(features.get("bm25_score", 0.0)) / 10.0, 1.0)
        score = float(np.clip(0.30 * coverage + 0.24 * overlap + 0.18 * jaccard + 0.18 * brand + 0.10 * bm25, 0, 1))

        if self.num_labels == 2:
            pos = 0.08 + 0.84 * score
            return np.array([[1.0 - pos, pos]], dtype=np.float32)

        # 3 sınıf: düşük skor alakasız, orta kısmen, yüksek çok alakalı.
        p2 = max(0.02, score ** 1.25)
        p0 = max(0.02, (1.0 - score) ** 1.15)
        p1 = max(0.02, 1.0 - abs(score - 0.50) * 1.55)
        arr = np.array([p0, p1, p2], dtype=np.float32)
        arr = arr / arr.sum()
        return arr.reshape(1, -1)

    def _hf_predict_proba(self, df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
        torch = getattr(self, "_torch", None)
        if self._hf_tokenizer is None or torch is None:
            return self._heuristic_proba(df, config)

        q_col = config.get("data", {}).get("query_column", "search_query")
        t_col = config.get("data", {}).get("product_text_column", "product_name")
        texts_a = df[q_col].fillna("").astype(str).tolist()
        texts_b = [build_product_text(row, config) or str(row.get(t_col, "")) for _, row in df.iterrows()]
        encoded = self._hf_tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return probs.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, train_df: pd.DataFrame, config: Dict[str, Any], val_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Hızlı ve offline-safe eğitim. HF fine-tune yerine feature classifier fallback kullanır."""
        t0 = time.perf_counter()
        train_df = self._ensure_product_columns(train_df, config)
        label_col = config.get("data", {}).get("label_column", "is_relevant")
        y = train_df[label_col].astype(int).clip(0, self.num_labels - 1).values if label_col in train_df.columns else np.zeros(len(train_df), dtype=int)

        X, _ = self._feature_pipeline.fit_transform(train_df)
        if len(np.unique(y)) < 2:
            clf = DummyClassifier(strategy="most_frequent")
        else:
            clf = make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=500, class_weight="balanced", multi_class="auto"),
            )
        clf.fit(X, y)
        self._classifier = clf
        self._mode = "feature_classifier"

        metrics: Dict[str, float] = {"training_time_seconds": float(time.perf_counter() - t0)}
        try:
            preds = self.predict(train_df, config)
            metrics["train_macro_f1"] = float(f1_score(y, preds, average="macro", zero_division=0))
        except Exception:
            metrics["train_macro_f1"] = 0.0
        return metrics

    def predict_proba(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        config = config or {}
        t0 = time.perf_counter()
        df = self._ensure_product_columns(df, config)

        if self._mode == "huggingface":
            probs = self._hf_predict_proba(df, config)
        elif self._classifier is not None:
            X = self._feature_pipeline.transform(df)
            raw = self._classifier.predict_proba(X)
            probs = np.zeros((len(df), self.num_labels), dtype=np.float32)
            classes = getattr(self._classifier, "classes_", None)
            if classes is None and hasattr(self._classifier, "steps"):
                classes = self._classifier.steps[-1][1].classes_
            for j, cls in enumerate(list(classes)):
                if int(cls) < self.num_labels:
                    probs[:, int(cls)] = raw[:, j]
            row_sums = probs.sum(axis=1, keepdims=True)
            missing = row_sums.squeeze() == 0
            if missing.any():
                probs[missing] = self._heuristic_proba(df[missing], config)
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
        else:
            probs = self._heuristic_proba(df, config)

        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return probs.astype(np.float32)

    def predict(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return np.argmax(self.predict_proba(df, config), axis=1)

    def predict_single(
        self,
        query: str,
        product: str,
        config: Optional[Dict[str, Any]] = None,
        brand: str = "",
        category: str = "",
        color: str = "",
        material: str = "",
    ) -> Dict[str, Any]:
        row = self._build_inference_df(query, product, config, brand, category, color, material)
        probs = self.predict_proba(row, config or {})[0]
        pred = int(np.argmax(probs))
        labels = self._labels()
        return {
            "predicted_label": pred,
            "predicted_class": labels[pred] if pred < len(labels) else str(pred),
            "confidence": float(probs[pred]),
            "probabilities": [float(x) for x in probs.tolist()],
            "model_mode": self._mode,
        }

    def evaluate(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        config = config or {}
        label_col = config.get("data", {}).get("label_column", "is_relevant")
        if label_col not in df.columns or len(df) == 0:
            return {"macro_f1": 0.0, "precision": 0.0, "recall": 0.0}
        y_true = df[label_col].astype(int).clip(0, self.num_labels - 1).values
        y_pred = self.predict(df, config)
        return {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "inference_latency_ms": float(self._last_inference_ms),
        }

    def get_attention_explanation(
        self,
        query: str,
        product: str,
        config: Optional[Dict[str, Any]] = None,
        brand: str = "",
        category: str = "",
        color: str = "",
        material: str = "",
    ) -> Dict[str, Any]:
        q_tokens = [t for t in str(query).lower().split() if t]
        context = " ".join([str(product), str(brand), str(category), str(color), str(material)]).lower()
        scored = []
        for tok in q_tokens:
            score = 1.0 if tok in context else 0.35
            scored.append({"token": tok, "score": score, "source": "query"})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"top_tokens": scored[:8], "method": "overlap_attention_fallback"}

    def load_onnx(self, path: str | Path) -> None:
        try:
            import onnxruntime as ort
            p = Path(path)
            model_file = p if p.is_file() else p / "model.onnx"
            if model_file.exists():
                self._onnx_session = ort.InferenceSession(str(model_file))
                self._mode = "onnx"
        except Exception as exc:
            print(f"[CrossEncoderModel] ONNX yüklenemedi: {exc}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "mode": self._mode,
            "artifact_note": "Feature fallback classifier; gerçek yarışma metriği değildir.",
        }
        write_json(path / "meta.json", meta)
        write_json(path / "config.json", {"num_labels": self.num_labels, "model_type": "deep-pipeline-cross-encoder"})
        if self._classifier is not None:
            with open(path / "classifier.pkl", "wb") as f:
                pickle.dump({"classifier": self._classifier, "feature_pipeline": self._feature_pipeline}, f)

    @classmethod
    def load(cls, path: str | Path) -> "CrossEncoderModel":
        path = Path(path)
        meta_path = path / "meta.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        model = cls(
            model_name=meta.get("model_name", "feature-fallback"),
            num_labels=int(meta.get("num_labels", 3)),
            max_length=int(meta.get("max_length", 256)),
            load_pretrained=False,
        )
        clf_path = path / "classifier.pkl"
        if clf_path.exists():
            with open(clf_path, "rb") as f:
                data = pickle.load(f)
            model._classifier = data.get("classifier")
            model._feature_pipeline = data.get("feature_pipeline", model._feature_pipeline)
            model._mode = meta.get("mode", "feature_classifier")
        return model
