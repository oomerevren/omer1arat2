"""Explain semantic confuser / dense hard-negative candidates."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src_kaggle.data.schema import SCHEMA


def dense_negative_subtype(row: dict) -> str:
    dense = float(row.get("dense_score", 0) or 0)
    lex = float(row.get("lexical_score", 0) or 0)
    if int(row.get("category_match_flag", 0)) and any(int(row.get(c, 0)) for c in ["color_conflict_flag", "material_conflict_flag", "gender_conflict_flag", "age_conflict_flag"]):
        return "dense_attribute_near_miss"
    if int(row.get("category_match_flag", 0)):
        return "dense_same_category_hard"
    if int(row.get("brand_match_flag", 0)):
        return "dense_same_brand_hard"
    if dense >= 0.50 and lex < 0.20:
        return "dense_semantic_confuser"
    return "dense_hard"


def write_semantic_confuser_report(negatives: pd.DataFrame, uncertain: pd.DataFrame, path: str | Path, max_examples: int = 30) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cand = pd.concat([negatives.assign(bucket="selected_negative"), uncertain.assign(bucket="uncertain_or_skipped")], ignore_index=True, sort=False)
    cand = cand[cand.get("source_pool", pd.Series(dtype=str)).astype(str).str.contains("dense", na=False)] if not cand.empty else cand
    lines = ["# Semantic Confuser Examples", "", "Dense kaynaklı adayların neden zor/riski yüksek olduğunu açıklayan rapor.", ""]
    if cand.empty:
        lines += ["Dense aday yok. `negative_mining.use_dense_pool=true` ve `dense_negatives_per_positive>0` ile yeniden çalıştırın."]
    else:
        cand = cand.sort_values(["safety_status", "dense_score"], ascending=[True, False]).head(max_examples)
        for _, r in cand.iterrows():
            lines += [
                f"## query={r.get(SCHEMA.query, '')} / item={r.get(SCHEMA.item_id, '')}",
                "",
                f"- title: {r.get(SCHEMA.title, '')}",
                f"- category: {r.get(SCHEMA.category, '')}",
                f"- brand: {r.get(SCHEMA.brand, '')}",
                f"- subtype: {r.get('dense_negative_subtype', dense_negative_subtype(r.to_dict()))}",
                f"- dense_score: {float(r.get('dense_score', 0) or 0):.4f}",
                f"- lexical_score: {float(r.get('lexical_score', 0) or 0):.4f}",
                f"- safety_status: {r.get('safety_status', '')}",
                f"- reasons: {r.get('safety_reasons', '')}",
                f"- why close: semantic embedding yakınlığı / kategori-marka-attribute sinyalleri.",
                f"- why maybe irrelevant: conflict flags gender={r.get('gender_conflict_flag', 0)}, age={r.get('age_conflict_flag', 0)}, color={r.get('color_conflict_flag', 0)}, material={r.get('material_conflict_flag', 0)}.",
                "",
            ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
