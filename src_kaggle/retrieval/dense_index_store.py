"""Persistent dense embedding/index store with versioned metadata."""
from __future__ import annotations

from pathlib import Path
import json
import pickle
from typing import Any

import numpy as np
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.retrieval.dense_retriever import DenseRetriever, _safe_slug


def dense_artifact_dir(root: str | Path, *, model_name: str, item_text_version: str, config_hash: str | None = None) -> Path:
    base = Path(root) / _safe_slug(model_name) / item_text_version
    return base / config_hash if config_hash else base


def save_dense_store(retriever: DenseRetriever, root: str | Path) -> Path:
    cfg_hash = retriever.metadata.get("config_hash", "unversioned")
    out_dir = dense_artifact_dir(root, model_name=retriever.model_name, item_text_version=retriever.item_text_version, config_hash=cfg_hash)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "item_embeddings.npy", retriever.embeddings)
    retriever.items[[SCHEMA.item_id]].to_csv(out_dir / "item_ids.csv", index=False)
    metadata = dict(retriever.metadata)
    metadata.update({
        "artifact_dir": str(out_dir),
        "item_embeddings": str(out_dir / "item_embeddings.npy"),
        "item_ids": str(out_dir / "item_ids.csv"),
        "index": str(out_dir / "index.pkl"),
    })
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    # Store the full retriever for API reload.  Real dense saves embeddings only + metadata; query encoder can be restored with load_sentence_model=True.
    retriever.save(out_dir / "index.pkl")
    return out_dir


def load_dense_store(index_path_or_dir: str | Path, *, load_sentence_model: bool = False, device: str | None = None) -> DenseRetriever:
    path = Path(index_path_or_dir)
    if path.is_dir():
        path = path / "index.pkl"
    return DenseRetriever.load(path, load_sentence_model=load_sentence_model, device=device)


def read_dense_metadata(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_file():
        path = path.parent
    meta = path / "metadata.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
