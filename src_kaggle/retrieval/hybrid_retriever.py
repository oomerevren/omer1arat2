"""Hybrid retrieval interface for negative mining and feature engineering."""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.data.attribute_parser import COLOR_VALUE_ALIASES, MATERIAL_VALUE_ALIASES, STYLE_VALUE_ALIASES
from src_kaggle.features.query_intent import build_query_intent_resources, extract_query_intent
from src_kaggle.retrieval.bm25_retriever import BM25Retriever
from src_kaggle.retrieval.dense_retriever import DenseRetriever
from src_kaggle.retrieval.embedding_builder import build_dense_retriever_from_config, dense_cfg_from_retrieval_cfg
from src_kaggle.retrieval.item_text_builder import prepare_items_for_retrieval


class HybridRetriever:
    def __init__(self, items: pd.DataFrame, bm25: BM25Retriever | None = None, dense: DenseRetriever | None = None, seed: int = 42, dense_text_version: str = "dense_v1"):
        self.items = prepare_items_for_retrieval(items, dense_text_version=dense_text_version) if SCHEMA.retrieval_text not in items.columns else items.reset_index(drop=True)
        self.bm25 = bm25
        self.dense = dense
        self.rng = np.random.default_rng(seed)
        self.intent_resources = build_query_intent_resources(self.items)

    @classmethod
    def build(cls, items: pd.DataFrame, *, use_bm25: bool = True, use_dense: bool = True, dense_model_name: str = "tfidf_svd_fallback", dense_backend: str | None = None, dense_text_version: str = "dense_v1", seed: int = 42, retrieval_cfg: dict | None = None) -> "HybridRetriever":
        if retrieval_cfg is not None:
            dense_cfg = dense_cfg_from_retrieval_cfg(retrieval_cfg)
            dense_text_version = dense_cfg.get("item_text_version", dense_text_version)
            use_dense = bool(dense_cfg.get("enabled", use_dense)) and "dense" in set(retrieval_cfg.get("enabled_retrievers", ["bm25", "dense"]))
            use_bm25 = "bm25" in set(retrieval_cfg.get("enabled_retrievers", ["bm25", "dense"]))
        prepared = prepare_items_for_retrieval(items, dense_text_version=dense_text_version)
        bm25 = BM25Retriever.fit(prepared) if use_bm25 else None
        if retrieval_cfg is not None and use_dense:
            dense = build_dense_retriever_from_config(prepared, retrieval_cfg)
        else:
            dense = DenseRetriever.fit(prepared, model_name=dense_model_name, backend=dense_backend or ("fallback_dense" if dense_model_name == "tfidf_svd_fallback" else "real_dense"), item_text_version=dense_text_version) if use_dense else None
        return cls(prepared, bm25=bm25, dense=dense, seed=seed, dense_text_version=dense_text_version)

    def lexical_nearest_pool(self, query: str, top_k: int = 50, exclude_items: set | None = None) -> pd.DataFrame:
        if self.bm25 is None:
            return pd.DataFrame()
        out = self.bm25.search(query, top_k, exclude_items)
        if not out.empty and "bm25_rank" not in out.columns:
            out["bm25_rank"] = range(1, len(out) + 1)
            out["bm25_score"] = out.get("score", 0.0)
        return out

    def dense_nearest_pool(self, query: str, top_k: int = 50, exclude_items: set | None = None) -> pd.DataFrame:
        if self.dense is None:
            return pd.DataFrame()
        return self.dense.search(query, top_k, exclude_items)

    def same_category_pool(self, categories: set[str] | list[str], top_k: int = 30, exclude_items: set | None = None) -> pd.DataFrame:
        exclude_items = exclude_items or set(); categories = set(map(str, categories))
        rows = self.items[self.items[SCHEMA.category].astype(str).isin(categories)].copy()
        rows = rows[~rows[SCHEMA.item_id].isin(exclude_items)].head(top_k)
        rows["source"] = "category_pool"; rows["score"] = 1.0
        return rows

    def same_brand_pool(self, brands: set[str] | list[str], top_k: int = 20, exclude_items: set | None = None) -> pd.DataFrame:
        exclude_items = exclude_items or set(); brands = set(map(str, brands))
        rows = self.items[self.items[SCHEMA.brand].astype(str).isin(brands)].copy()
        rows = rows[~rows[SCHEMA.item_id].isin(exclude_items)].head(top_k)
        rows["source"] = "brand_pool"; rows["score"] = 1.0
        return rows

    def random_far_pool(self, query: str, top_k: int = 30, exclude_items: set | None = None, max_bm25: float = 0.01) -> pd.DataFrame:
        exclude_items = exclude_items or set()
        scores = self.bm25.scores(query) if self.bm25 is not None else np.zeros(len(self.items))
        eligible = [i for i, row in self.items.iterrows() if row[SCHEMA.item_id] not in exclude_items and scores[i] <= max_bm25]
        if not eligible:
            eligible = [i for i, row in self.items.iterrows() if row[SCHEMA.item_id] not in exclude_items]
        chosen = self.rng.choice(eligible, size=min(top_k, len(eligible)), replace=False) if eligible else []
        rows = self.items.iloc[list(chosen)].copy() if len(chosen) else pd.DataFrame()
        if not rows.empty:
            rows["source"] = "random_far_pool"; rows["score"] = [float(scores[i]) for i in chosen]
        return rows

    def attribute_similar_pool(self, query: str, top_k: int = 30, exclude_items: set | None = None) -> pd.DataFrame:
        exclude_items = exclude_items or set()
        intent = extract_query_intent(query, self.intent_resources)
        colors = set(filter(None, str(intent.get("detected_color_candidates", "")).split("|")))
        colors |= {COLOR_VALUE_ALIASES.get(c, c) for c in list(colors)}
        mats = set(filter(None, str(intent.get("detected_material_candidates", "")).split("|")))
        mats |= {MATERIAL_VALUE_ALIASES.get(m, m) for m in list(mats)}
        styles = set(filter(None, str(intent.get("detected_style_candidates", "")).split("|")))
        styles |= {STYLE_VALUE_ALIASES.get(s, s) for s in list(styles)}
        rows = self.items[~self.items[SCHEMA.item_id].isin(exclude_items)].copy()
        if rows.empty:
            return rows
        def sim(row):
            s = 0
            s += int(bool(colors) and not colors.isdisjoint(set(str(row.get(SCHEMA.color_value, "")).split("|"))))
            s += int(bool(mats) and not mats.isdisjoint(set(str(row.get(SCHEMA.material_value, "")).split("|"))))
            s += int(bool(styles) and not styles.isdisjoint(set(str(row.get(SCHEMA.style_value, "")).split("|"))))
            return s
        rows["score"] = rows.apply(sim, axis=1)
        rows = rows[rows["score"] > 0].sort_values("score", ascending=False).head(top_k)
        rows["source"] = "attribute_similar_pool"
        return rows

    def hybrid_search(self, query: str, top_k: int = 50, bm25_k: int = 50, dense_k: int = 50, exclude_items: set | None = None) -> pd.DataFrame:
        bm25 = self.lexical_nearest_pool(query, bm25_k, exclude_items)
        dense = self.dense_nearest_pool(query, dense_k, exclude_items)
        if bm25.empty and dense.empty:
            return pd.DataFrame()
        bm = {r[SCHEMA.item_id]: (i + 1, float(r.get("score", 0))) for i, (_, r) in enumerate(bm25.iterrows())} if not bm25.empty else {}
        de = {r[SCHEMA.item_id]: (i + 1, float(r.get("dense_score", r.get("score", 0)))) for i, (_, r) in enumerate(dense.iterrows())} if not dense.empty else {}
        ids = list(dict.fromkeys(list(bm.keys()) + list(de.keys())))
        rows = self.items[self.items[SCHEMA.item_id].isin(ids)].copy()
        rows["bm25_rank"] = rows[SCHEMA.item_id].map(lambda x: bm.get(x, (0, 0))[0])
        rows["bm25_score"] = rows[SCHEMA.item_id].map(lambda x: bm.get(x, (0, 0))[1])
        rows["dense_rank"] = rows[SCHEMA.item_id].map(lambda x: de.get(x, (0, 0))[0])
        rows["dense_score"] = rows[SCHEMA.item_id].map(lambda x: de.get(x, (0, 0))[1])
        rows["source"] = rows.apply(lambda r: "bm25|dense" if r.bm25_rank and r.dense_rank else ("bm25" if r.bm25_rank else "dense"), axis=1)
        rows["hybrid_score"] = rows["bm25_score"] + rows["dense_score"]
        return rows.sort_values(["hybrid_score", "dense_score"], ascending=False).head(top_k)

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "HybridRetriever":
        with Path(path).open("rb") as f:
            return pickle.load(f)
