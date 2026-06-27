from __future__ import annotations

import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.retrieval.hybrid_retriever import HybridRetriever


def build_retrieval_features(df: pd.DataFrame, retriever: HybridRetriever | None = None, top_k: int = 100) -> pd.DataFrame:
    rows=[]
    cache: dict[str, dict] = {}
    for _, r in df.iterrows():
        item_id = r.get(SCHEMA.item_id)
        query = r.get(SCHEMA.query, "")
        key = str(query)
        if retriever is not None and key not in cache:
            bm25 = retriever.lexical_nearest_pool(query, top_k)
            dense = retriever.dense_nearest_pool(query, top_k)
            bm = {x[SCHEMA.item_id]: (rank+1, float(x.get("score",0))) for rank, (_, x) in enumerate(bm25.iterrows())} if not bm25.empty else {}
            de = {x[SCHEMA.item_id]: (rank+1, float(x.get("score",0))) for rank, (_, x) in enumerate(dense.iterrows())} if not dense.empty else {}
            cache[key] = {"bm25": bm, "dense": de}
        bm_rank=bm_score=de_rank=de_score=0.0
        if retriever is not None:
            bm = cache[key]["bm25"].get(item_id)
            de = cache[key]["dense"].get(item_id)
            if bm: bm_rank, bm_score = bm
            if de: de_rank, de_score = de
        else:
            bm_score = float(r.get("bm25_score", r.get("lexical_score", 0)) or 0)
            de_score = float(r.get("dense_score", 0) or 0)
        bm_hit = int(bm_rank > 0); de_hit = int(de_rank > 0)
        rank_gap = float(bm_rank or (top_k + 1)) - float(de_rank or (top_k + 1))
        attr_conflict = int(r.get("color_conflict_flag", 0) or r.get("material_conflict_flag", 0) or r.get("gender_conflict_flag", 0) or r.get("age_conflict_flag", 0))
        rows.append({
            "retrieval_bm25_score": bm_score,
            "retrieval_dense_score": de_score,
            "retrieval_dense_score_real": de_score if (retriever is not None and getattr(getattr(retriever, "dense", None), "is_real_dense", False)) else float(r.get("dense_score", de_score) or 0),
            "retrieval_bm25_rank": bm_rank,
            "retrieval_dense_rank": de_rank,
            "retrieval_dense_rank_real": de_rank,
            "retrieval_dense_rank_percentile": (float(de_rank) / float(top_k)) if de_rank else 0.0,
            "retrieval_in_bm25_topk_flag": bm_hit,
            "retrieval_in_dense_topk_flag": de_hit,
            "retrieval_dense_only_hit_flag": int(de_hit and not bm_hit),
            "retrieval_bm25_dense_overlap_flag": int(bm_hit and de_hit),
            "retrieval_dense_bm25_rank_gap": rank_gap,
            "retrieval_hybrid_consensus_flag": int(bm_hit and de_hit and abs(rank_gap) <= max(3, top_k * 0.1)),
            "retrieval_rank_agreement_flag": int(bm_hit and de_hit),
            "retrieval_lexical_dense_gap": float(bm_score) - float(de_score),
            "retrieval_low_lexical_high_dense_flag": int(float(bm_score) < 0.15 and float(de_score) >= 0.35),
            "retrieval_high_lexical_low_dense_flag": int(float(bm_score) >= 0.35 and float(de_score) < 0.15),
            "retrieval_category_match_dense_high_flag": int(r.get("category_match_flag", 0) and float(de_score) >= 0.35),
            "retrieval_attribute_conflict_dense_high_flag": int(attr_conflict and float(de_score) >= 0.35),
            "retrieval_same_category_pool_flag": int(r.get("category_match_flag", 0) or r.get("source_pool", "") == "category_filtered_pool"),
            "retrieval_same_brand_pool_flag": int(r.get("brand_match_flag", 0) or r.get("source_pool", "") == "brand_aware_pool"),
        })
    return pd.DataFrame(rows, index=df.index)
