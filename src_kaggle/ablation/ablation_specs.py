"""Ablation experiment specifications for Kaggle War Mode.

Specs are intentionally declarative: each row should change one interpretable
component.  The runner applies these specs to the base config and records OOF
metrics or a clear not-run reason.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    experiment_name: str
    category: str  # feature | negative | retrieval | model | threshold | dense
    changed_component: str
    variant_description: str
    toggles: dict[str, Any]
    risk_note: str = ""
    requires_transformer: bool = False
    requires_real_dense: bool = False
    requires_negative_rebuild: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def feature_group_specs() -> list[AblationSpec]:
    remove = [
        ("no_lexical", "use_lexical_features"),
        ("no_category", "use_category_features"),
        ("no_brand", "use_brand_features"),
        ("no_attribute", "use_attribute_features"),
        ("no_gender_age", "use_gender_age_features"),
        ("no_query_features", "use_query_features"),
        ("no_retrieval", "use_retrieval_features"),
        ("no_semantic", "use_semantic_features"),
        ("no_metadata", "use_metadata_features"),
    ]
    specs = [AblationSpec("feat_all_features", "tab_all_features", "feature", "all_features", "All tabular feature groups enabled", {"feature_mode": "all"})]
    for name, flag in remove:
        specs.append(AblationSpec(f"feat_{name}", f"tab_{name}", "feature", flag, f"All features except {flag}", {"feature_disable": [flag]}))
    combos = [
        ("only_core_features", ["use_lexical_features", "use_category_features", "use_brand_features", "use_attribute_features", "use_gender_age_features"]),
        ("only_lexical_category_brand", ["use_lexical_features", "use_category_features", "use_brand_features"]),
        ("only_attribute_gender_age", ["use_attribute_features", "use_gender_age_features"]),
        ("lexical_plus_retrieval", ["use_lexical_features", "use_retrieval_features"]),
        ("attribute_plus_semantic", ["use_attribute_features", "use_gender_age_features", "use_semantic_features"]),
    ]
    for name, enabled in combos:
        specs.append(AblationSpec(f"feat_{name}", f"tab_{name}", "feature", name, f"Only feature groups: {', '.join(enabled)}", {"feature_only": enabled}))
    return specs


def negative_mining_specs() -> list[AblationSpec]:
    def nm(name: str, changed: str, desc: str, counts: dict[str, int], extra: dict[str, Any] | None = None, dense: bool = False):
        return AblationSpec(f"neg_{name}", f"tab_{name}_neg", "negative", changed, desc, {"negative_counts": counts, **(extra or {})}, requires_negative_rebuild=True, requires_real_dense=dense)
    zero = {"easy": 0, "same_category": 0, "same_brand": 0, "lexical": 0, "attribute_conflict": 0, "dense": 0}
    specs = [
        nm("easy_only", "easy", "Only random/easy negatives", {**zero, "easy": 2}),
        nm("easy_plus_same_category", "same_category", "Easy + same-category negatives", {**zero, "easy": 1, "same_category": 2}),
        nm("easy_plus_same_brand", "same_brand", "Easy + same-brand negatives", {**zero, "easy": 1, "same_brand": 2}),
        nm("easy_plus_attribute_conflict", "attribute_conflict", "Easy + attribute-conflict negatives", {**zero, "easy": 1, "attribute_conflict": 2}),
        nm("easy_plus_lexical_confusing", "lexical_confusing", "Easy + lexical confusing negatives", {**zero, "easy": 1, "lexical": 2}),
        nm("easy_plus_dense_hard", "dense_hard", "Easy + dense semantic hard negatives", {**zero, "easy": 1, "dense": 2}, {"use_dense_pool": True}, dense=True),
        nm("full_negative_mix", "full_mix", "Full negative mix including dense", {"easy": 1, "same_category": 1, "same_brand": 1, "lexical": 1, "attribute_conflict": 1, "dense": 2}, {"use_dense_pool": True}, dense=True),
        nm("full_negative_mix_without_dense", "full_mix_no_dense", "Full negative mix without dense", {"easy": 1, "same_category": 1, "same_brand": 1, "lexical": 1, "attribute_conflict": 1, "dense": 0}, {"use_dense_pool": False}),
        nm("strict_false_negative_filter", "false_negative_filter", "Strict false-negative filter", {"easy": 1, "same_category": 1, "same_brand": 1, "lexical": 1, "attribute_conflict": 1, "dense": 1}, {"filter_profile": "strict", "use_dense_pool": True}, dense=True),
        nm("moderate_false_negative_filter", "false_negative_filter", "Moderate false-negative filter", {"easy": 1, "same_category": 1, "same_brand": 1, "lexical": 1, "attribute_conflict": 1, "dense": 1}, {"filter_profile": "moderate", "use_dense_pool": True}, dense=True),
    ]
    return specs


def retrieval_specs() -> list[AblationSpec]:
    return [
        AblationSpec("ret_no_retrieval_features", "tab_no_retrieval_features", "retrieval", "retrieval_features", "No retrieval-derived features", {"feature_disable": ["use_retrieval_features"]}),
        AblationSpec("ret_bm25_only", "tab_bm25_only_retrieval", "retrieval", "bm25_only", "BM25 retrieval features only", {"retrieval_mode": "bm25_only"}),
        AblationSpec("ret_dense_only", "tab_dense_only_retrieval", "retrieval", "dense_only", "Dense retrieval features only", {"retrieval_mode": "dense_only"}, requires_real_dense=True),
        AblationSpec("ret_hybrid", "tab_hybrid_retrieval", "retrieval", "hybrid", "BM25 + dense hybrid retrieval features", {"retrieval_mode": "hybrid"}, requires_real_dense=True),
        AblationSpec("ret_rank_off", "tab_retrieval_rank_off", "retrieval", "rank_features", "Retrieval score features without rank signals", {"retrieval_feature_submode": "score_only"}),
        AblationSpec("ret_score_off", "tab_retrieval_score_off", "retrieval", "score_features", "Retrieval rank features without score signals", {"retrieval_feature_submode": "rank_only"}),
        AblationSpec("ret_overlap_off", "tab_retrieval_overlap_off", "retrieval", "overlap_agreement", "Retrieval features without overlap/agreement flags", {"retrieval_feature_submode": "no_overlap"}),
    ]


def special_signal_specs() -> list[AblationSpec]:
    return [
        AblationSpec("sig_no_attribute_conflict", "tab_no_attribute_conflict", "feature", "attribute_conflict_signals", "Attribute exact/conflict signals off", {"special_disable": ["attribute_conflict"]}),
        AblationSpec("sig_no_gender_conflict", "tab_no_gender_conflict", "feature", "gender_conflict", "Gender conflict signals off", {"feature_disable": ["use_gender_age_features"]}),
        AblationSpec("sig_no_age_conflict", "tab_no_age_conflict", "feature", "age_conflict", "Age conflict signals off", {"feature_disable": ["use_gender_age_features"]}),
        AblationSpec("sig_no_brand_contradiction", "tab_no_brand_contradiction", "feature", "brand_contradiction", "Brand contradiction/exact signals off", {"feature_disable": ["use_brand_features"]}),
        AblationSpec("sig_no_size_number", "tab_no_size_number", "feature", "size_number", "Size/number signals off", {"special_disable": ["size_number"]}),
    ]


def model_family_specs() -> list[AblationSpec]:
    return [
        AblationSpec("model_tabular_baseline", "tabular_baseline", "model", "tabular", "Tabular model only", {"model_type": "tabular"}),
        AblationSpec("model_sklearn_text_baseline", "sklearn_text_baseline", "model", "sklearn_text", "Sklearn TF-IDF pair-text cross-encoder baseline", {"model_type": "cross_encoder", "ce_backend": "sklearn_text"}),
        AblationSpec("model_transformer_ce_best", "transformer_ce_best", "model", "transformer_ce", "Real transformer cross-encoder; requires GPU/deps", {"model_type": "cross_encoder", "ce_backend": "transformers"}, requires_transformer=True),
        AblationSpec("model_tabular_plus_sklearn_text", "tabular_plus_sklearn_text", "model", "blend_tab_sklearn", "Blend tabular + sklearn_text OOF", {"blend": ["tabular", "sklearn_text"]}),
        AblationSpec("model_tabular_plus_transformer", "tabular_plus_transformer", "model", "blend_tab_transformer", "Blend tabular + transformer OOF", {"blend": ["tabular", "transformer"]}, requires_transformer=True),
        AblationSpec("model_tabular_plus_transformer_plus_dense", "tabular_plus_transformer_plus_dense_signals", "model", "blend_tab_transformer_dense", "Tabular dense signals + transformer blend", {"blend": ["tabular_dense", "transformer"]}, requires_transformer=True, requires_real_dense=True),
        AblationSpec("model_best_single", "best_single_model", "model", "best_single", "Select best single model from completed OOFs", {"selection": "best_single"}),
        AblationSpec("model_best_balanced_blend", "best_balanced_blend", "model", "best_blend", "Select best balanced blend from completed OOFs", {"selection": "best_blend"}),
    ]


def threshold_specs() -> list[AblationSpec]:
    return [
        AblationSpec("thr_fixed_05", "threshold_fixed_05", "threshold", "fixed_05", "Fixed threshold 0.5", {"threshold_strategy": "fixed_05"}),
        AblationSpec("thr_oof_best", "threshold_oof_best", "threshold", "oof_best", "OOF best global threshold", {"threshold_strategy": "oof_best"}),
        AblationSpec("thr_class0_protective", "threshold_class0_protective", "threshold", "class0_protective", "Higher threshold to protect class 0", {"threshold_strategy": "class0_protective"}),
        AblationSpec("thr_class1_protective", "threshold_class1_protective", "threshold", "class1_protective", "Lower threshold to protect class 1", {"threshold_strategy": "class1_protective"}),
        AblationSpec("thr_stable_midpoint", "threshold_stable_midpoint", "threshold", "stable_midpoint", "Midpoint of stable near-optimal threshold plateau", {"threshold_strategy": "stable_midpoint"}),
        AblationSpec("thr_segment_analysis", "threshold_segment_analysis_only", "threshold", "segment_threshold", "Segment threshold analysis only, not final decision", {"threshold_strategy": "segment_analysis_only"}, risk_note="Segment thresholds are high overfit risk unless validated by private-LB simulation."),
    ]


def dense_specs() -> list[AblationSpec]:
    return [
        AblationSpec("dense_no_dense_anywhere", "no_dense_anywhere", "dense", "dense_off", "Dense retrieval/negatives/features disabled", {"dense_mode": "off"}),
        AblationSpec("dense_features_only", "dense_features_only", "dense", "dense_features", "Dense retrieval features on, dense negatives off", {"dense_mode": "features_only"}, requires_real_dense=True),
        AblationSpec("dense_hard_negatives_only", "dense_hard_negatives_only", "dense", "dense_negatives", "Dense hard negatives on, dense feature group off", {"dense_mode": "negatives_only"}, requires_real_dense=True, requires_negative_rebuild=True),
        AblationSpec("dense_features_plus_negatives", "dense_features_plus_dense_negatives", "dense", "dense_features_and_negatives", "Dense retrieval features and dense hard negatives together", {"dense_mode": "features_plus_negatives"}, requires_real_dense=True, requires_negative_rebuild=True),
        AblationSpec("dense_text_v1", "dense_text_v1", "dense", "item_text_version", "Dense item text version v1", {"dense_text_version": "dense_v1"}, requires_real_dense=True),
        AblationSpec("dense_text_v2", "dense_text_v2", "dense", "item_text_version", "Dense item text version v2", {"dense_text_version": "dense_v2"}, requires_real_dense=True),
        AblationSpec("dense_backend_A", "dense_backend_A_multilingual_minilm", "dense", "dense_backend_A", "Default multilingual MiniLM sentence-transformer", {"dense_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}, requires_real_dense=True),
        AblationSpec("dense_backend_B", "dense_backend_B_turkish_candidate", "dense", "dense_backend_B", "Alternative Turkish/multilingual sentence model candidate", {"dense_model_name": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"}, requires_real_dense=True),
    ]


def all_ablation_specs() -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    for fn in [feature_group_specs, negative_mining_specs, retrieval_specs, special_signal_specs, model_family_specs, threshold_specs, dense_specs]:
        specs.extend(fn())
    return specs
