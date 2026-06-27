"""BM25 vs dense retrieval evaluation and segment reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.query_intent import add_query_intent_features, build_query_intent_resources
from src_kaggle.retrieval.hybrid_retriever import HybridRetriever


def _idset(df: pd.DataFrame) -> set:
    return set(df[SCHEMA.item_id].tolist()) if df is not None and not df.empty else set()


def _segment(row: pd.Series) -> str:
    if int(row.get("is_brand_heavy", 0)): return "brand_heavy"
    if int(row.get("is_attribute_heavy", 0)): return "attribute_heavy"
    if int(row.get("has_gender_token", 0)): return "gender_cue"
    if int(row.get("is_short_query", 0)): return "short_query"
    if int(row.get("is_long_query", 0)): return "long_tail_query"
    return "general"


def evaluate_retrieval(queries: pd.DataFrame, retriever: HybridRetriever, *, top_k: int = 20) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "query_segment" not in queries.columns:
        resources = build_query_intent_resources(retriever.items)
        queries = add_query_intent_features(queries.copy(), resources)
        queries["query_segment"] = queries.apply(_segment, axis=1)
    rows = []
    for _, qrow in queries.drop_duplicates(SCHEMA.term_id).iterrows():
        q = str(qrow[SCHEMA.query])
        bm25 = retriever.lexical_nearest_pool(q, top_k)
        dense = retriever.dense_nearest_pool(q, top_k)
        bm_ids, de_ids = _idset(bm25), _idset(dense)
        union = bm_ids | de_ids
        dense_only = de_ids - bm_ids
        # Dense behavior explainability against query metadata if available.
        pos_cat = str(qrow.get(SCHEMA.category, ""))
        pos_brand = str(qrow.get(SCHEMA.brand, ""))
        cat_match = float(dense[SCHEMA.category].astype(str).eq(pos_cat).mean()) if pos_cat and not dense.empty else 0.0
        brand_match = float(dense[SCHEMA.brand].astype(str).eq(pos_brand).mean()) if pos_brand and not dense.empty else 0.0
        rows.append({
            SCHEMA.term_id: qrow.get(SCHEMA.term_id),
            SCHEMA.query: q,
            "query_segment": qrow.get("query_segment", "general"),
            "bm25_hits": len(bm_ids),
            "dense_hits": len(de_ids),
            "overlap_count": len(bm_ids & de_ids),
            "bm25_dense_overlap_jaccard": len(bm_ids & de_ids) / max(1, len(union)),
            "dense_only_count": len(dense_only),
            "dense_topk_category_match_rate": cat_match,
            "dense_topk_brand_match_rate": brand_match,
            "dense_backend": retriever.dense.backend if retriever.dense is not None else "none",
            "semantic_backend_active": bool(retriever.dense and retriever.dense.is_real_dense),
        })
    comp = pd.DataFrame(rows)
    seg = comp.groupby("query_segment").agg({
        "bm25_dense_overlap_jaccard": "mean",
        "dense_only_count": "mean",
        "dense_topk_category_match_rate": "mean",
        "dense_topk_brand_match_rate": "mean",
        "term_id": "count",
    }).rename(columns={"term_id": "query_count"}).reset_index() if not comp.empty else pd.DataFrame()
    report = {
        "top_k": top_k,
        "dense_backend": retriever.dense.backend if retriever.dense is not None else "none",
        "semantic_backend_active": bool(retriever.dense and retriever.dense.is_real_dense),
        "overall_overlap_jaccard_mean": float(comp["bm25_dense_overlap_jaccard"].mean()) if not comp.empty else 0.0,
        "overall_dense_only_mean": float(comp["dense_only_count"].mean()) if not comp.empty else 0.0,
        "segments": {str(r["query_segment"]): {k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v) for k, v in r.items() if k != "query_segment"} for _, r in seg.iterrows()},
        "leakage_note": "For OOF, dense index text comes from public item universe; dense hard negative selection must be regenerated per fold using train positives only for positive exclusion.",
    }
    return comp, report


def write_retrieval_evaluation(comp: pd.DataFrame, report: dict[str, Any], *, comparison_csv: str | Path, segment_json: str | Path) -> None:
    comparison_csv = Path(comparison_csv); segment_json = Path(segment_json)
    comparison_csv.parent.mkdir(parents=True, exist_ok=True); segment_json.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(comparison_csv, index=False)
    segment_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
