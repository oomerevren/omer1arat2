"""Rule-based error taxonomy for OOF/validation mistakes."""
from __future__ import annotations

from typing import Any
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_utils import normalize_text, token_set

ERROR_TYPES = [
    "brand_mismatch", "category_mismatch", "attribute_mismatch", "gender_conflict",
    "age_group_conflict", "title_ambiguity", "query_ambiguity", "synonym_vocabulary_issue",
    "typo_spelling_issue", "missing_attribute_issue", "false_negative_sampling_issue",
    "retrieval_source_issue", "threshold_error", "calibration_error",
]


def _int(row: pd.Series, col: str) -> int:
    try: return int(float(row.get(col, 0) or 0))
    except Exception: return 0


def assign_error_tags(row: pd.Series, threshold: float = 0.5) -> list[str]:
    tags: list[str] = []
    q = row.get(SCHEMA.query, ""); title = row.get(SCHEMA.title, "")
    qset, tset = token_set(q), token_set(title)
    proba = float(row.get("proba", 0.5) or 0.5)
    label = _int(row, SCHEMA.label); pred = _int(row, "pred_best_threshold") if "pred_best_threshold" in row else int(proba >= threshold)

    if _int(row, "brand_contradiction_flag") or (_int(row, "has_brand_token") and not _int(row, "brand_exact_match") and label != pred):
        tags.append("brand_mismatch")
    if _int(row, "cat_token_overlap_count") == 0 and _int(row, "has_category_token") and label != pred:
        tags.append("category_mismatch")
    if _int(row, "attr_conflict_count") or _int(row, "attr_color_conflict_flag") or _int(row, "attr_material_conflict_flag"):
        tags.append("attribute_mismatch")
    if _int(row, "gender_conflict_flag"):
        tags.append("gender_conflict")
    if _int(row, "age_conflict_flag"):
        tags.append("age_group_conflict")
    if len(tset) <= 2 or (qset and len(qset & tset) == 0 and label != pred):
        tags.append("title_ambiguity")
    if _int(row, "is_short_query") or len(qset) <= 2:
        tags.append("query_ambiguity")
    if _int(row, "possible_typo_or_ambiguous"):
        tags.append("typo_spelling_issue")
    if normalize_text(row.get(SCHEMA.attributes, "")) == "" or _int(row, "attr_color_missing_when_query_has_color") or _int(row, "attr_material_missing_when_query_has_material"):
        tags.append("missing_attribute_issue")
    if label == 1 and str(row.get("negative_type", "")) not in {"", "positive"}:
        tags.append("false_negative_sampling_issue")
    if str(row.get("source_pool", "")) in {"lexical_nearest_pool", "dense_nearest_pool", "category_filtered_pool", "brand_aware_pool"} and label != pred:
        tags.append("retrieval_source_issue")
    if abs(proba - threshold) <= 0.05:
        tags.append("threshold_error")
    if (label == 0 and proba >= 0.9) or (label == 1 and proba <= 0.1):
        tags.append("calibration_error")
    if not tags:
        if qset and tset and len(qset & tset) == 0:
            tags.append("synonym_vocabulary_issue")
        else:
            tags.append("query_ambiguity")
    return sorted(set(tags))


def action_suggestions(error_counts: dict[str, int]) -> list[str]:
    suggestions=[]
    if error_counts.get("attribute_mismatch",0) or error_counts.get("missing_attribute_issue",0):
        suggestions.append("Attribute parser/feature coverage genişlet; color/material/style conflict feature ağırlığını ve attribute-conflict negatives oranını kontrol et.")
    if error_counts.get("brand_mismatch",0):
        suggestions.append("Same-brand hard negative oranını artır ve brand contradiction feature'larını segment bazlı incele.")
    if error_counts.get("gender_conflict",0) or error_counts.get("age_group_conflict",0):
        suggestions.append("Gender/age cue kurallarını güçlendir; conflict içeren örneklerde threshold/class-0 F1 kontrol et.")
    if error_counts.get("category_mismatch",0):
        suggestions.append("Category parsing/hierarchy feature ekle; same-category hard negative havuzunu iyileştir.")
    if error_counts.get("retrieval_source_issue",0):
        suggestions.append("Retrieval pool kalitesini analiz et; dense pool ve BM25/dense rank feature'larını yeniden kalibre et.")
    if error_counts.get("threshold_error",0):
        suggestions.append("Threshold sensitivity yüksek olabilir; global threshold çevresinde class 0/1 trade-off analiz et.")
    if error_counts.get("calibration_error",0):
        suggestions.append("Probability calibration veya ensemble blend ağırlıklarını OOF üzerinden yeniden değerlendir.")
    if error_counts.get("false_negative_sampling_issue",0):
        suggestions.append("False-negative safety layer sıkılaştır; uncertain_candidate exclusion threshold'u yükselt.")
    return suggestions or ["Belirgin tek hata kaynağı yok; segment bazlı OOF ve model disagreement raporlarını incele."]
