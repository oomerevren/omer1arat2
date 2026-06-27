"""Competition-grade, explainable negative mining for Kaggle War Mode."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_kaggle.data.attribute_parser import add_attribute_features
from src_kaggle.data.negative_filters import assess_candidate_safety, conflict_flags
from src_kaggle.data.negative_pools import CandidatePool
from src_kaggle.retrieval.hybrid_retriever import HybridRetriever
from src_kaggle.retrieval.semantic_confuser_analysis import dense_negative_subtype, write_semantic_confuser_report
from src_kaggle.data.pair_builder import build_full_item_text
from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.query_intent import build_query_intent_resources, extract_query_intent
from src_kaggle.data.io import write_table


@dataclass(frozen=True)
class NegativeMiningConfig:
    enabled: bool = True
    seed: int = 42
    mode: str = "global"  # global | fold_aware
    fold_column: str = "fold"
    active_fold: int | None = None
    exclude_uncertain: bool = True
    max_candidates_per_query: int = 250
    easy_negatives_per_positive: int = 1
    same_category_negatives_per_positive: int = 1
    same_brand_negatives_per_positive: int = 1
    lexical_negatives_per_positive: int = 1
    attribute_conflict_negatives_per_positive: int = 1
    dense_negatives_per_positive: int = 0
    use_dense_pool: bool = False
    dense_top_k: int = 100
    retrieval_cfg: dict[str, Any] | None = None
    false_negative_thresholds: dict[str, float] | None = None

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> "NegativeMiningConfig":
        thresholds = cfg.get("false_negative_thresholds") or {
            "very_high_lexical": 0.92,
            "variant_lexical": 0.82,
            "hard_lexical": 0.35,
            "very_high_dense": 0.94,
        }
        return NegativeMiningConfig(
            enabled=bool(cfg.get("enabled", True)),
            seed=int(cfg.get("seed", 42)),
            mode=str(cfg.get("mode", "global")),
            fold_column=str(cfg.get("fold_column", "fold")),
            active_fold=cfg.get("active_fold"),
            exclude_uncertain=bool(cfg.get("exclude_uncertain", True)),
            max_candidates_per_query=int(cfg.get("max_candidates_per_query", 250)),
            easy_negatives_per_positive=int(cfg.get("easy_negatives_per_positive", 1)),
            same_category_negatives_per_positive=int(cfg.get("same_category_negatives_per_positive", 1)),
            same_brand_negatives_per_positive=int(cfg.get("same_brand_negatives_per_positive", 1)),
            lexical_negatives_per_positive=int(cfg.get("lexical_negatives_per_positive", 1)),
            attribute_conflict_negatives_per_positive=int(cfg.get("attribute_conflict_negatives_per_positive", 1)),
            dense_negatives_per_positive=int(cfg.get("dense_negatives_per_positive", 0)),
            use_dense_pool=bool(cfg.get("use_dense_pool", False)),
            dense_top_k=int(cfg.get("dense_top_k", cfg.get("max_candidates_per_query", 100))),
            retrieval_cfg=cfg.get("retrieval_cfg"),
            false_negative_thresholds=cfg.get("dense_false_negative_thresholds") or thresholds,
        )


def _prepare_items(items: pd.DataFrame) -> pd.DataFrame:
    out = add_attribute_features(items) if SCHEMA.normalized_attribute_text not in items.columns else items.copy()
    return build_full_item_text(out).reset_index(drop=True)


def _positive_training_view(positive_pairs: pd.DataFrame, cfg: NegativeMiningConfig) -> pd.DataFrame:
    if cfg.mode == "fold_aware" and cfg.active_fold is not None and cfg.fold_column in positive_pairs.columns:
        return positive_pairs[positive_pairs[cfg.fold_column] != cfg.active_fold].reset_index(drop=True)
    return positive_pairs.reset_index(drop=True)


def _pool_plan(cfg: NegativeMiningConfig) -> list[tuple[str, str, int]]:
    return [
        ("easy", "random_far_pool", cfg.easy_negatives_per_positive),
        ("same_category", "category_filtered_pool", cfg.same_category_negatives_per_positive),
        ("same_brand", "brand_aware_pool", cfg.same_brand_negatives_per_positive),
        ("lexical_confusing", "lexical_nearest_pool", cfg.lexical_negatives_per_positive),
        ("attribute_conflict", "attribute_conflict_pool", cfg.attribute_conflict_negatives_per_positive),
        ("dense_hard", "dense_nearest_pool", cfg.dense_negatives_per_positive),
    ]


def _candidate_rows(pool: CandidatePool, query: str, source_pool: str, positives_for_term: set, pos_categories: set, pos_brands: set, k: int, dense_retriever: HybridRetriever | None = None) -> pd.DataFrame:
    request_k = max(k * 12, 30)
    if source_pool == "random_far_pool":
        return pool.random_far(query, request_k, positives_for_term)
    if source_pool == "category_filtered_pool":
        return pool.same_category(pos_categories, query, request_k, positives_for_term)
    if source_pool == "brand_aware_pool":
        return pool.same_brand(pos_brands, query, request_k, positives_for_term)
    if source_pool == "attribute_conflict_pool":
        return pool.attribute_conflict(query, request_k, positives_for_term)
    if source_pool == "dense_nearest_pool" and dense_retriever is not None:
        dense = dense_retriever.dense_nearest_pool(query, request_k, positives_for_term)
        if dense.empty:
            return dense
        lex_scores = pool.lexical_scores(query)
        idx_by_item = {row[SCHEMA.item_id]: i for i, row in pool.items.iterrows()}
        dense["lexical_score"] = dense[SCHEMA.item_id].map(lambda x: float(lex_scores[idx_by_item[x]]) if x in idx_by_item else 0.0)
        return dense
    return pool.top_lexical(query, request_k, positives_for_term)


def mine_negatives(positive_pairs: pd.DataFrame, items: pd.DataFrame, config: NegativeMiningConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.enabled:
        return pd.DataFrame(), pd.DataFrame()

    positives = _positive_training_view(positive_pairs, config)
    items_prepared = _prepare_items(items)
    pool = CandidatePool.from_items(items_prepared, seed=config.seed)
    resources = build_query_intent_resources(items_prepared)
    dense_retriever = None
    if config.use_dense_pool and config.dense_negatives_per_positive > 0:
        retrieval_cfg = dict(config.retrieval_cfg or {})
        if retrieval_cfg:
            retrieval_cfg.setdefault("enabled_retrievers", ["bm25", "dense"])
        dense_retriever = HybridRetriever.build(items_prepared, seed=config.seed, retrieval_cfg=retrieval_cfg or {"dense": {"backend": "fallback_dense", "model_name": "tfidf_svd_fallback"}})
        if dense_retriever.dense is not None and not dense_retriever.dense.is_real_dense:
            print("[WARN] Dense negative mining fallback_dense ile çalışıyor; gerçek semantic hard negative aktif değildir.")
    rng = np.random.default_rng(config.seed)

    pos_by_term = positives.groupby(SCHEMA.term_id)[SCHEMA.item_id].apply(set).to_dict()
    meta_by_term = positives.groupby(SCHEMA.term_id).agg({SCHEMA.category: lambda x: set(map(str, x)), SCHEMA.brand: lambda x: set(map(str, x))}).to_dict("index")
    query_by_term = positives.groupby(SCHEMA.term_id)[SCHEMA.query].first().to_dict()

    records: list[dict[str, Any]] = []
    uncertain_records: list[dict[str, Any]] = []
    selected_pairs: set[tuple[Any, Any]] = set()

    for term_id, group in positives.groupby(SCHEMA.term_id, sort=True):
        query = query_by_term[term_id]
        q_intent = extract_query_intent(query, resources)
        positives_for_term = set(pos_by_term.get(term_id, set()))
        pos_categories = set(meta_by_term.get(term_id, {}).get(SCHEMA.category, set()))
        pos_brands = set(meta_by_term.get(term_id, {}).get(SCHEMA.brand, set()))
        n_pos = len(group)

        for neg_type, source_pool, per_pos in _pool_plan(config):
            target_n = int(per_pos) * n_pos
            if target_n <= 0:
                continue
            if source_pool == "dense_nearest_pool" and not config.use_dense_pool:
                continue
            candidates = _candidate_rows(pool, query, source_pool, positives_for_term, pos_categories, pos_brands, target_n, dense_retriever=dense_retriever)
            if candidates.empty:
                continue
            # deterministic but avoids always taking identical tie order across pools
            candidates = candidates.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
            taken = 0
            for _, cand in candidates.iterrows():
                item_id = cand[SCHEMA.item_id]
                pair_key = (term_id, item_id)
                if pair_key in selected_pairs or item_id in positives_for_term:
                    continue
                lexical_score = float(cand.get("lexical_score", cand.get("bm25_score", 0.0)) or 0.0)
                dense_score = float(cand.get("dense_score", np.nan)) if pd.notna(cand.get("dense_score", np.nan)) else None
                category_match = int(str(cand.get(SCHEMA.category, "")) in pos_categories)
                brand_match = int(str(cand.get(SCHEMA.brand, "")) in pos_brands)
                conflict = conflict_flags(q_intent, cand.to_dict())
                decision = assess_candidate_safety(
                    term_id=term_id,
                    item_id=item_id,
                    candidate=cand.to_dict(),
                    positive_items_for_term=positives_for_term,
                    lexical_score=lexical_score,
                    dense_score=dense_score,
                    category_match_flag=category_match,
                    brand_match_flag=brand_match,
                    conflict=conflict,
                    thresholds=config.false_negative_thresholds or {},
                )
                dense_component = 0.0 if dense_score is None else max(0.0, min(1.0, dense_score))
                hardness = min(1.0, 0.35 * lexical_score + 0.35 * dense_component + 0.13 * category_match + 0.10 * brand_match + 0.07 * min(1, sum(conflict.values())))
                rec = cand.to_dict()
                rec.update(q_intent)
                rec.update(conflict)
                rec.update({
                    SCHEMA.term_id: term_id,
                    SCHEMA.query: query,
                    SCHEMA.label: 0,
                    "negative_type": neg_type,
                    "source_pool": source_pool,
                    "dense_negative_subtype": dense_negative_subtype({**cand.to_dict(), **conflict, "category_match_flag": category_match, "brand_match_flag": brand_match, "dense_score": dense_score or 0, "lexical_score": lexical_score}) if source_pool == "dense_nearest_pool" else neg_type,
                    "safety_status": decision.status,
                    "safety_score": decision.safety_score,
                    "hardness_score": float(hardness),
                    "lexical_score": lexical_score,
                    "dense_score": np.nan if dense_score is None else dense_score,
                    "dense_backend": getattr(getattr(dense_retriever, "dense", None), "backend", "none") if source_pool == "dense_nearest_pool" else "none",
                    "category_match_flag": category_match,
                    "brand_match_flag": brand_match,
                    "safety_reasons": "|".join(decision.reasons),
                })
                if decision.status in {"uncertain_candidate", "skipped_due_to_false_negative_risk"}:
                    uncertain_records.append(rec)
                    if config.exclude_uncertain or decision.status == "skipped_due_to_false_negative_risk":
                        continue
                records.append(rec)
                selected_pairs.add(pair_key)
                taken += 1
                if taken >= target_n:
                    break

    negatives = pd.DataFrame.from_records(records)
    uncertain = pd.DataFrame.from_records(uncertain_records)
    return negatives.reset_index(drop=True), uncertain.reset_index(drop=True)


def build_augmented_training_set(positives: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    pos = positives.copy()
    pos["negative_type"] = "positive"
    pos["source_pool"] = "official_training_pairs"
    pos["safety_status"] = "known_positive"
    pos["safety_score"] = 1.0
    pos["hardness_score"] = 0.0
    pos["lexical_score"] = np.nan
    pos["dense_score"] = np.nan
    for col in ["category_match_flag", "brand_match_flag", "gender_conflict_flag", "age_conflict_flag", "color_conflict_flag", "material_conflict_flag"]:
        if col not in pos.columns:
            pos[col] = 0
    if negatives.empty:
        return pos
    cols = list(dict.fromkeys(list(pos.columns) + list(negatives.columns)))
    return pd.concat([pos.reindex(columns=cols), negatives.reindex(columns=cols)], ignore_index=True)


def make_negative_mining_report(negatives: pd.DataFrame, uncertain: pd.DataFrame, output_path: str) -> dict[str, Any]:
    total = int(len(negatives))
    problem_segments = {}
    if total and {"is_attribute_heavy", "is_brand_heavy", "is_category_heavy"}.issubset(negatives.columns):
        grouped = negatives.groupby(["is_attribute_heavy", "is_brand_heavy", "is_category_heavy"]).size().sort_values(ascending=False).head(20)
        problem_segments = {
            f"attribute_heavy={int(k[0])},brand_heavy={int(k[1])},category_heavy={int(k[2])}": int(v)
            for k, v in grouped.items()
        }

    report = {
        "output_path": output_path,
        "total_negatives": total,
        "uncertain_excluded_or_flagged": int(len(uncertain)),
        "uncertain_ratio_vs_selected": float(len(uncertain) / max(1, total + len(uncertain))),
        "avg_negatives_per_query": float(negatives.groupby(SCHEMA.term_id).size().mean()) if total else 0.0,
        "negative_type_distribution": {str(k): int(v) for k, v in negatives.get("negative_type", pd.Series(dtype=str)).value_counts().to_dict().items()},
        "source_pool_distribution": {str(k): int(v) for k, v in negatives.get("source_pool", pd.Series(dtype=str)).value_counts().to_dict().items()},
        "dense_negative_subtype_distribution": {str(k): int(v) for k, v in negatives.get("dense_negative_subtype", pd.Series(dtype=str)).value_counts().to_dict().items()},
        "safety_status_distribution": {str(k): int(v) for k, v in pd.concat([negatives.get("safety_status", pd.Series(dtype=str)), uncertain.get("safety_status", pd.Series(dtype=str))]).value_counts().to_dict().items()},
        "dense_score_summary": {str(k): float(v) for k, v in negatives.get("dense_score", pd.Series(dtype=float)).dropna().describe().to_dict().items()} if total and "dense_score" in negatives else {},
        "same_category_ratio": float(negatives.get("category_match_flag", pd.Series(dtype=float)).mean()) if total else 0.0,
        "same_brand_ratio": float(negatives.get("brand_match_flag", pd.Series(dtype=float)).mean()) if total else 0.0,
        "lexical_score_summary": {str(k): float(v) for k, v in negatives.get("lexical_score", pd.Series(dtype=float)).describe().to_dict().items()} if total else {},
        "hardness_score_summary": {str(k): float(v) for k, v in negatives.get("hardness_score", pd.Series(dtype=float)).describe().to_dict().items()} if total else {},
        "category_negative_density_top20": {str(k): int(v) for k, v in negatives.get(SCHEMA.category, pd.Series(dtype=str)).value_counts().head(20).to_dict().items()} if total else {},
        "problem_query_segments": problem_segments,
    }
    return report


def write_negative_reports(report: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path); md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines = ["# Negative Mining Report", "", f"- total_negatives: {report['total_negatives']}", f"- uncertain_excluded_or_flagged: {report['uncertain_excluded_or_flagged']}", f"- avg_negatives_per_query: {report['avg_negatives_per_query']:.3f}", f"- same_category_ratio: {report['same_category_ratio']:.3f}", f"- same_brand_ratio: {report['same_brand_ratio']:.3f}", "", "## Negative type distribution", ""]
    for k, v in report.get("negative_type_distribution", {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Source pool distribution", ""]
    for k, v in report.get("source_pool_distribution", {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Dense subtype distribution", ""]
    for k, v in report.get("dense_negative_subtype_distribution", {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Safety status distribution", ""]
    for k, v in report.get("safety_status_distribution", {}).items():
        lines.append(f"- {k}: {v}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_negative_mining_and_write(*, positives: pd.DataFrame, items: pd.DataFrame, cfg: NegativeMiningConfig, output_path: str, uncertain_path: str, report_json_path: str, report_md_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    negatives, uncertain = mine_negatives(positives, items, cfg)
    augmented = build_augmented_training_set(positives, negatives)
    write_table(augmented, output_path)
    write_table(uncertain, uncertain_path)
    report = make_negative_mining_report(negatives, uncertain, output_path)
    write_negative_reports(report, report_json_path, report_md_path)
    return augmented, uncertain, report
