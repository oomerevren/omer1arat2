"""Dense semantic retrieval with explicit real/fallback backends.

Important contract:
- ``backend='real_dense'`` never silently falls back. Missing dependencies/model/GPU
  constraints raise a clear RuntimeError.
- ``backend='fallback_dense'`` is an explicit TF-IDF+SVD dense-like debug backend.
  Reports and metadata label it as fallback so semantic hard-negative claims are
  not confused with real embedding retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import pickle
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src_kaggle.data.schema import SCHEMA
from src_kaggle.retrieval.item_text_builder import build_query_dense_text, prepare_items_for_retrieval

REAL_BACKEND = "real_dense"
FALLBACK_BACKEND = "fallback_dense"
LEGACY_FALLBACK_NAME = "tfidf_svd_fallback"


def _safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))[:160]


def config_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


@dataclass
class DenseRetriever:
    items: pd.DataFrame
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    backend: str = REAL_BACKEND
    item_text_version: str = "dense_v1"
    query_text_version: str = "raw"
    normalize_embeddings: bool = True
    vectorizer: object | None = None
    svd: object | None = None
    embeddings: np.ndarray | None = None
    sentence_model: object | None = None
    text_col: str = SCHEMA.dense_text
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_real_dense(self) -> bool:
        return self.backend == REAL_BACKEND

    @classmethod
    def fit(
        cls,
        items: pd.DataFrame,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        *,
        backend: str = REAL_BACKEND,
        n_components: int = 128,
        text_col: str = SCHEMA.dense_text,
        item_text_version: str = "dense_v1",
        query_text_version: str = "raw",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
        require_gpu: bool = False,
    ) -> "DenseRetriever":
        if model_name == LEGACY_FALLBACK_NAME:
            backend = FALLBACK_BACKEND
        if backend not in {REAL_BACKEND, FALLBACK_BACKEND}:
            raise ValueError(f"Dense backend must be one of real_dense/fallback_dense, got {backend!r}")
        prepared = prepare_items_for_retrieval(items, dense_text_version=item_text_version) if text_col not in items.columns else items.reset_index(drop=True).copy()
        texts = prepared[text_col].fillna("").astype(str).tolist()
        build_cfg = {
            "backend": backend, "model_name": model_name, "item_text_version": item_text_version,
            "query_text_version": query_text_version, "item_count": len(prepared), "text_col": text_col,
            "normalize_embeddings": normalize_embeddings,
        }
        if backend == REAL_BACKEND:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Dense retrieval backend=real_dense seçildi ama sentence-transformers yüklü değil. "
                    "Sessiz fallback yasaktır. Ya bağımlılığı kurun ya da config'te backend=fallback_dense yapın."
                ) from e
            if require_gpu:
                try:
                    import torch  # type: ignore
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA not available")
                except Exception as e:
                    raise RuntimeError("Dense retrieval backend=real_dense için require_gpu=true ama GPU yok/torch erişilemiyor.") from e
            try:
                sm = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
                emb = sm.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=normalize_embeddings)
            except Exception as e:
                raise RuntimeError(
                    f"Real dense model yüklenemedi/encode edilemedi: {model_name}. "
                    "Fallback çalıştırılmadı; explicit backend=fallback_dense gerekir."
                ) from e
            arr = np.asarray(emb, dtype=np.float32)
            md = {**build_cfg, "embedding_dim": int(arr.shape[1]) if arr.ndim == 2 else 0, "built_at": datetime.now(timezone.utc).isoformat(), "config_hash": config_hash(build_cfg), "semantic_backend_active": True}
            return cls(prepared, model_name, backend, item_text_version, query_text_version, normalize_embeddings, None, None, arr, sm, text_col, md)

        vectorizer = TfidfVectorizer(max_features=200_000, ngram_range=(1, 2), min_df=1)
        X = vectorizer.fit_transform(texts)
        if min(X.shape) > 2:
            n_comp = max(2, min(n_components, min(X.shape) - 1))
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            arr = svd.fit_transform(X)
        else:
            svd = None
            arr = X.toarray()
        arr = normalize(arr).astype(np.float32) if normalize_embeddings else np.asarray(arr, dtype=np.float32)
        md = {**build_cfg, "model_name": LEGACY_FALLBACK_NAME, "embedding_dim": int(arr.shape[1]) if arr.ndim == 2 else 0, "built_at": datetime.now(timezone.utc).isoformat(), "config_hash": config_hash(build_cfg), "semantic_backend_active": False, "warning": "fallback_dense is not real semantic retrieval"}
        return cls(prepared, LEGACY_FALLBACK_NAME, FALLBACK_BACKEND, item_text_version, query_text_version, normalize_embeddings, vectorizer, svd, arr, None, text_col, md)

    def encode_query(self, query: str) -> np.ndarray:
        qtext = build_query_dense_text(query, self.query_text_version)
        if self.sentence_model is not None:
            emb = self.sentence_model.encode([qtext], normalize_embeddings=self.normalize_embeddings, show_progress_bar=False)
            return np.asarray(emb, dtype=np.float32)[0]
        if self.vectorizer is None:
            raise RuntimeError("Fallback dense query encoder is missing vectorizer; rebuild/load a valid index.")
        X = self.vectorizer.transform([qtext])
        arr = self.svd.transform(X) if self.svd is not None else X.toarray()
        arr = normalize(arr).astype(np.float32) if self.normalize_embeddings else np.asarray(arr, dtype=np.float32)
        return arr[0]

    def search(self, query: str, top_k: int = 50, exclude_items: set | None = None) -> pd.DataFrame:
        exclude_items = exclude_items or set()
        if self.embeddings is None or len(self.items) == 0:
            return pd.DataFrame()
        q = self.encode_query(query)
        scores = self.embeddings @ q
        order = np.argsort(-scores)
        rows = []
        rank = 0
        for idx in order:
            row = self.items.iloc[idx].copy()
            if row[SCHEMA.item_id] in exclude_items:
                continue
            rank += 1
            row["source"] = "dense"
            row["dense_backend"] = self.backend
            row["score"] = float(scores[idx])
            row["dense_score"] = float(scores[idx])
            row["rank"] = rank
            row["dense_rank"] = rank
            rows.append(row)
            if len(rows) >= top_k:
                break
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        obj = self
        if self.sentence_model is not None:
            obj = DenseRetriever(
                items=self.items, model_name=self.model_name, backend=self.backend,
                item_text_version=self.item_text_version, query_text_version=self.query_text_version,
                normalize_embeddings=self.normalize_embeddings, embeddings=self.embeddings,
                sentence_model=None, text_col=self.text_col, metadata=self.metadata,
            )
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(path: str | Path, *, load_sentence_model: bool = False, device: str | None = None) -> "DenseRetriever":
        with Path(path).open("rb") as f:
            obj: DenseRetriever = pickle.load(f)
        if load_sentence_model and obj.backend == REAL_BACKEND and obj.sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                obj.sentence_model = SentenceTransformer(obj.model_name, device=device) if device else SentenceTransformer(obj.model_name)
            except Exception as e:
                raise RuntimeError(f"Saved real dense index loaded but query encoder model cannot be restored: {obj.model_name}") from e
        return obj
