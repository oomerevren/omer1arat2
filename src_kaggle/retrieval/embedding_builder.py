"""High-level dense embedding/index builder from Kaggle retrieval config."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src_kaggle.retrieval.dense_index_store import save_dense_store
from src_kaggle.retrieval.dense_retriever import DenseRetriever, REAL_BACKEND, FALLBACK_BACKEND


def dense_cfg_from_retrieval_cfg(retrieval_cfg: dict[str, Any]) -> dict[str, Any]:
    dense = dict(retrieval_cfg.get("dense", {}) or {})
    # Backward compatible legacy keys.
    if "model_name" not in dense and retrieval_cfg.get("dense_model_name"):
        dense["model_name"] = retrieval_cfg.get("dense_model_name")
    dense.setdefault("enabled", "dense" in set(retrieval_cfg.get("enabled_retrievers", ["bm25", "dense"])))
    dense.setdefault("backend", REAL_BACKEND)
    dense.setdefault("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    dense.setdefault("item_text_version", "dense_v1")
    dense.setdefault("query_text_version", "raw")
    dense.setdefault("top_k", retrieval_cfg.get("dense_top_k", 100))
    dense.setdefault("persist_index", retrieval_cfg.get("persist_indices", True))
    dense.setdefault("artifact_root", retrieval_cfg.get("dense_index_root", "artifacts/retrieval/dense"))
    dense.setdefault("batch_size", 64)
    dense.setdefault("normalize_embeddings", True)
    dense.setdefault("device", None)
    dense.setdefault("require_gpu", False)
    return dense


def build_dense_retriever_from_config(items: pd.DataFrame, retrieval_cfg: dict[str, Any]) -> DenseRetriever | None:
    dense = dense_cfg_from_retrieval_cfg(retrieval_cfg)
    if not dense.get("enabled", True):
        return None
    return DenseRetriever.fit(
        items,
        model_name=dense["model_name"],
        backend=dense.get("backend", REAL_BACKEND),
        item_text_version=dense.get("item_text_version", "dense_v1"),
        query_text_version=dense.get("query_text_version", "raw"),
        batch_size=int(dense.get("batch_size", 64)),
        normalize_embeddings=bool(dense.get("normalize_embeddings", True)),
        device=dense.get("device"),
        require_gpu=bool(dense.get("require_gpu", False)),
    )


def build_and_optionally_persist_dense(items: pd.DataFrame, retrieval_cfg: dict[str, Any]) -> tuple[DenseRetriever | None, str | None]:
    dense_cfg = dense_cfg_from_retrieval_cfg(retrieval_cfg)
    retriever = build_dense_retriever_from_config(items, retrieval_cfg)
    artifact_dir = None
    if retriever is not None and dense_cfg.get("persist_index", True):
        artifact_dir = str(save_dense_store(retriever, dense_cfg.get("artifact_root", "artifacts/retrieval/dense")))
    return retriever, artifact_dir
