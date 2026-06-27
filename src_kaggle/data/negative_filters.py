"""False-negative safety filters for Kaggle negative mining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src_kaggle.data.schema import SCHEMA


@dataclass(frozen=True)
class SafetyDecision:
    status: str  # safe_negative | hard_negative | uncertain_candidate | skipped_due_to_false_negative_risk
    safety_score: float
    reasons: tuple[str, ...]


def _norm(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def conflict_flags(query_intent: dict, candidate: dict) -> dict[str, int]:
    """Detect explainable query-vs-item conflicts."""
    gender_conflict = 0
    age_conflict = 0
    color_conflict = 0
    material_conflict = 0

    detected_gender = set(filter(None, str(query_intent.get("detected_gender_candidates", "")).split("|")))
    item_gender = _norm(candidate.get(SCHEMA.gender))
    if detected_gender and item_gender:
        if "unisex" not in detected_gender and "unisex" not in item_gender:
            if ("female" in detected_gender and "erkek" in item_gender) or ("male" in detected_gender and "kad" in item_gender):
                gender_conflict = 1

    detected_age = set(filter(None, str(query_intent.get("detected_age_candidates", "")).split("|")))
    item_age = _norm(candidate.get(SCHEMA.age_group))
    if detected_age and item_age:
        if "baby" in detected_age and "bebek" not in item_age:
            age_conflict = 1
        if "child" in detected_age and not any(x in item_age for x in ("çocuk", "cocuk", "bebek")):
            age_conflict = 1

    q_colors = set(filter(None, str(query_intent.get("detected_color_candidates", "")).split("|")))
    item_colors = set(filter(None, str(candidate.get(SCHEMA.color_value, "")).split("|")))
    if q_colors and item_colors and q_colors.isdisjoint(item_colors):
        color_conflict = 1

    q_materials = set(filter(None, str(query_intent.get("detected_material_candidates", "")).split("|")))
    item_materials = set(filter(None, str(candidate.get(SCHEMA.material_value, "")).split("|")))
    if q_materials and item_materials and q_materials.isdisjoint(item_materials):
        material_conflict = 1

    return {
        "gender_conflict_flag": gender_conflict,
        "age_conflict_flag": age_conflict,
        "color_conflict_flag": color_conflict,
        "material_conflict_flag": material_conflict,
    }


def assess_candidate_safety(
    *,
    term_id,
    item_id,
    candidate: dict,
    positive_items_for_term: set,
    lexical_score: float,
    dense_score: float | None,
    category_match_flag: int,
    brand_match_flag: int,
    conflict: dict[str, int],
    thresholds: dict,
) -> SafetyDecision:
    reasons: list[str] = []
    if item_id in positive_items_for_term:
        return SafetyDecision("uncertain_candidate", 0.0, ("known_positive_for_term",))

    high_lex = lexical_score >= float(thresholds.get("very_high_lexical", 0.92))
    high_dense = dense_score is not None and dense_score >= float(thresholds.get("very_high_dense", 0.94))
    dense_plus_lex = dense_score is not None and dense_score >= float(thresholds.get("dense_plus_lexical_high", 0.88)) and lexical_score >= float(thresholds.get("hard_lexical", 0.35))
    variant_dense = dense_score is not None and dense_score >= float(thresholds.get("semantic_variant_risk", 0.90))
    no_conflict = not any(conflict.values())
    strong_conflict = any(conflict.values())

    if high_lex and category_match_flag and no_conflict:
        reasons.append("too_close_lexical_same_category_no_conflict")
    if high_dense and no_conflict:
        reasons.append("too_close_dense_no_conflict")
    if dense_plus_lex and no_conflict:
        reasons.append("dense_plus_lexical_high_no_conflict")
    if category_match_flag and brand_match_flag and (lexical_score >= float(thresholds.get("variant_lexical", 0.82)) or variant_dense) and no_conflict:
        reasons.append("possible_variant_same_brand_category")

    if "known_positive_for_term" in reasons:
        return SafetyDecision("skipped_due_to_false_negative_risk", 0.0, tuple(reasons))
    if reasons and len(reasons) >= 2:
        return SafetyDecision("skipped_due_to_false_negative_risk", 0.10, tuple(reasons))
    if reasons:
        return SafetyDecision("uncertain_candidate", 0.25, tuple(reasons))

    conflict_count = sum(conflict.values())
    if dense_score is not None and dense_score >= float(thresholds.get("dense_hard_min", 0.35)):
        score = 0.70 + 0.07 * min(conflict_count, 3)
        return SafetyDecision("hard_negative", min(score, 0.96), ("dense_hard_but_safe" if strong_conflict else "dense_hard_low_fn_risk",))
    if lexical_score >= float(thresholds.get("hard_lexical", 0.35)) or category_match_flag or brand_match_flag:
        score = 0.65 + 0.08 * min(conflict_count, 3)
        return SafetyDecision("hard_negative", min(score, 0.95), ("hard_but_safe",))

    return SafetyDecision("safe_negative", 0.9, ("far_or_low_similarity",))
