"""Apply ablation toggles to a Kaggle config copy."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

ALL_FEATURE_FLAGS = [
    "use_lexical_features", "use_category_features", "use_brand_features", "use_attribute_features",
    "use_gender_age_features", "use_query_features", "use_retrieval_features", "use_semantic_features",
    "use_metadata_features",
]

NEG_COUNT_KEY = {
    "easy": "easy_negatives_per_positive",
    "same_category": "same_category_negatives_per_positive",
    "same_brand": "same_brand_negatives_per_positive",
    "lexical": "lexical_negatives_per_positive",
    "attribute_conflict": "attribute_conflict_negatives_per_positive",
    "dense": "dense_negatives_per_positive",
}


def _ensure(cfg: dict, *keys: str) -> dict:
    cur = cfg
    for k in keys:
        cur = cur.setdefault(k, {})
    return cur


def apply_toggles(base_cfg: dict[str, Any], toggles: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    feat = _ensure(cfg, "feature_engineering")
    retrieval = _ensure(cfg, "retrieval")
    dense = _ensure(cfg, "retrieval", "dense")
    nm = _ensure(cfg, "negative_mining")
    val = _ensure(cfg, "validation_framework")

    if toggles.get("feature_mode") == "all":
        for f in ALL_FEATURE_FLAGS: feat[f] = True
    for f in toggles.get("feature_disable", []) or []:
        feat[f] = False
    if toggles.get("feature_only"):
        enabled = set(toggles["feature_only"])
        for f in ALL_FEATURE_FLAGS: feat[f] = f in enabled

    if toggles.get("retrieval_mode"):
        mode = toggles["retrieval_mode"]
        feat["use_retrieval_features"] = mode != "none"
        if mode == "bm25_only":
            retrieval["enabled_retrievers"] = ["bm25"]; dense["enabled"] = False
        elif mode == "dense_only":
            retrieval["enabled_retrievers"] = ["dense"]; dense["enabled"] = True
        elif mode == "hybrid":
            retrieval["enabled_retrievers"] = ["bm25", "dense"]; dense["enabled"] = True

    # The current feature builder emits all retrieval feature subcolumns together;
    # submodes are recorded in config/report for controlled interpretation.
    if toggles.get("retrieval_feature_submode"):
        feat["retrieval_feature_submode"] = toggles["retrieval_feature_submode"]

    if toggles.get("model_type"):
        val["model_type"] = toggles["model_type"]
    if toggles.get("ce_backend"):
        ce = _ensure(cfg, "modeling", "cross_encoder")
        ce["backend"] = toggles["ce_backend"]
        val["model"] = ce
    elif val.get("model_type") == "tabular":
        val["model"] = cfg.get("modeling", {}).get("tabular", val.get("model", {}))

    if toggles.get("negative_counts"):
        for k, v in toggles["negative_counts"].items():
            nm[NEG_COUNT_KEY[k]] = int(v)
        nm["use_dense_pool"] = int(toggles["negative_counts"].get("dense", 0)) > 0 or bool(toggles.get("use_dense_pool", False))
    if "use_dense_pool" in toggles:
        nm["use_dense_pool"] = bool(toggles["use_dense_pool"])
    if toggles.get("filter_profile"):
        if toggles["filter_profile"] == "strict":
            nm["exclude_uncertain"] = True
            nm["dense_false_negative_thresholds"] = {"very_high_dense": 0.90, "dense_plus_lexical_high": 0.82, "semantic_variant_risk": 0.86, "hard_lexical": 0.30, "variant_lexical": 0.76}
        elif toggles["filter_profile"] == "moderate":
            nm["exclude_uncertain"] = True
            nm["dense_false_negative_thresholds"] = {"very_high_dense": 0.96, "dense_plus_lexical_high": 0.91, "semantic_variant_risk": 0.93, "hard_lexical": 0.40, "variant_lexical": 0.86}

    if toggles.get("dense_mode"):
        mode = toggles["dense_mode"]
        if mode == "off":
            dense["enabled"] = False; retrieval["enabled_retrievers"] = ["bm25"]; feat["use_retrieval_features"] = True; nm["use_dense_pool"] = False; nm["dense_negatives_per_positive"] = 0
        elif mode == "features_only":
            dense["enabled"] = True; retrieval["enabled_retrievers"] = ["bm25", "dense"]; feat["use_retrieval_features"] = True; nm["use_dense_pool"] = False; nm["dense_negatives_per_positive"] = 0
        elif mode == "negatives_only":
            dense["enabled"] = True; retrieval["enabled_retrievers"] = ["bm25", "dense"]; feat["use_retrieval_features"] = False; nm["use_dense_pool"] = True; nm["dense_negatives_per_positive"] = max(1, int(nm.get("dense_negatives_per_positive", 2)))
        elif mode == "features_plus_negatives":
            dense["enabled"] = True; retrieval["enabled_retrievers"] = ["bm25", "dense"]; feat["use_retrieval_features"] = True; nm["use_dense_pool"] = True; nm["dense_negatives_per_positive"] = max(1, int(nm.get("dense_negatives_per_positive", 2)))
    if toggles.get("dense_text_version"):
        dense["item_text_version"] = toggles["dense_text_version"]
    if toggles.get("dense_model_name"):
        dense["model_name"] = toggles["dense_model_name"]; retrieval["dense_model_name"] = toggles["dense_model_name"]

    if toggles.get("threshold_strategy"):
        _ensure(cfg, "threshold_optimization")["strategy"] = toggles["threshold_strategy"]

    cfg.setdefault("ablation", {})["applied_toggles"] = toggles
    return cfg
