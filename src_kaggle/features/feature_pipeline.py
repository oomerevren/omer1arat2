"""Modular feature engineering pipeline for Kaggle War Mode."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src_kaggle.data.attribute_parser import add_attribute_features
from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.attribute_features import build_attribute_features
from src_kaggle.features.brand_features import build_brand_features
from src_kaggle.features.category_features import build_category_features
from src_kaggle.features.feature_utils import ensure_numeric
from src_kaggle.features.gender_age_features import build_gender_age_features
from src_kaggle.features.lexical_features import build_lexical_features
from src_kaggle.features.metadata_features import build_metadata_features
from src_kaggle.features.query_features import build_query_features
from src_kaggle.features.retrieval_features import build_retrieval_features
from src_kaggle.features.semantic_features import build_semantic_features
from src_kaggle.features.query_intent import add_query_intent_features, build_query_intent_resources
from src_kaggle.retrieval.hybrid_retriever import HybridRetriever
from src_kaggle.retrieval.item_text_builder import prepare_items_for_retrieval


@dataclass
class FeaturePipelineConfig:
    use_lexical_features: bool = True
    use_category_features: bool = True
    use_brand_features: bool = True
    use_attribute_features: bool = True
    use_gender_age_features: bool = True
    use_query_features: bool = True
    use_retrieval_features: bool = True
    use_semantic_features: bool = True
    use_metadata_features: bool = True
    retrieval_top_k: int = 100

    @staticmethod
    def from_dict(cfg: dict[str, Any] | None) -> "FeaturePipelineConfig":
        cfg = cfg or {}
        known = {f.name for f in FeaturePipelineConfig.__dataclass_fields__.values()}  # type: ignore
        return FeaturePipelineConfig(**{k: v for k, v in cfg.items() if k in known})


@dataclass
class FeaturePipelineResult:
    features: pd.DataFrame
    feature_groups: dict[str, list[str]]
    feature_catalog: dict[str, Any]


class FeaturePipeline:
    def __init__(self, config: FeaturePipelineConfig | None = None, items: pd.DataFrame | None = None, retriever: HybridRetriever | None = None):
        self.config = config or FeaturePipelineConfig()
        self.items = prepare_items_for_retrieval(items) if items is not None and not items.empty else items
        self.retriever = retriever
        if self.retriever is None and self.items is not None and self.config.use_retrieval_features:
            self.retriever = HybridRetriever.build(self.items, dense_model_name="tfidf_svd_fallback", dense_backend="fallback_dense")

    def _prepare_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if SCHEMA.normalized_attribute_text not in out.columns and SCHEMA.attributes in out.columns:
            out = add_attribute_features(out)
        if not {"query_length_tokens", "has_brand_token", "detected_color_candidates"}.issubset(out.columns):
            resources = build_query_intent_resources(self.items) if self.items is not None else None
            out = add_query_intent_features(out, resources)
        if SCHEMA.full_item_text not in out.columns:
            out[SCHEMA.full_item_text] = out[[SCHEMA.title, SCHEMA.category, SCHEMA.brand, SCHEMA.normalized_attribute_text]].fillna("").astype(str).agg(" ".join, axis=1)
        return out

    def transform(self, df: pd.DataFrame) -> FeaturePipelineResult:
        pairs = self._prepare_pairs(df)
        frames: list[pd.DataFrame] = []
        groups: dict[str, list[str]] = {}

        def add(group: str, frame: pd.DataFrame):
            frame = ensure_numeric(frame)
            frames.append(frame)
            groups[group] = list(frame.columns)

        if self.config.use_lexical_features:
            add("lexical", build_lexical_features(pairs))
        if self.config.use_category_features:
            add("category", build_category_features(pairs))
        if self.config.use_brand_features:
            add("brand", build_brand_features(pairs))
        if self.config.use_attribute_features:
            add("attribute", build_attribute_features(pairs))
        if self.config.use_gender_age_features:
            add("gender_age", build_gender_age_features(pairs))
        if self.config.use_query_features:
            add("query", build_query_features(pairs, self.items))
        if self.config.use_retrieval_features:
            add("retrieval", build_retrieval_features(pairs, self.retriever, self.config.retrieval_top_k))
        if self.config.use_semantic_features:
            add("semantic", build_semantic_features(pairs))
        if self.config.use_metadata_features:
            add("metadata", build_metadata_features(pairs))

        features = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
        features = ensure_numeric(features)
        catalog = {
            "total_features": int(features.shape[1]),
            "groups": {g: {"count": len(cols), "features": cols} for g, cols in groups.items()},
            "null_rates": features.isna().mean().to_dict(),
            "constant_features": [c for c in features.columns if features[c].nunique(dropna=False) <= 1],
        }
        return FeaturePipelineResult(features=features, feature_groups=groups, feature_catalog=catalog)

    def explain_row(self, df: pd.DataFrame, row_idx: int = 0) -> dict[str, dict[str, float]]:
        result = self.transform(df.iloc[[row_idx]])
        row = result.features.iloc[0].to_dict()
        return {g: {c: row[c] for c in cols if c in row} for g, cols in result.feature_groups.items()}


def write_feature_reports(result: FeaturePipelineResult, catalog_path: str | Path, summary_path: str | Path, sample_rows: pd.DataFrame | None = None) -> None:
    catalog_path = Path(catalog_path); summary_path = Path(summary_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(result.feature_catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Feature Summary", "", f"Total features: {result.features.shape[1]}", "", "## Groups", ""]
    for g, cols in result.feature_groups.items():
        lines.append(f"- {g}: {len(cols)}")
    lines += ["", "## Constant features", ""]
    for c in result.feature_catalog.get("constant_features", [])[:100]:
        lines.append(f"- {c}")
    lines += ["", "## Highest null rates", ""]
    nulls = sorted(result.feature_catalog.get("null_rates", {}).items(), key=lambda x: x[1], reverse=True)[:50]
    for c, v in nulls:
        lines.append(f"- {c}: {v:.4f}")
    if sample_rows is not None and not sample_rows.empty:
        lines += ["", "## Sample feature rows", "", sample_rows.head(10).to_markdown(index=False)]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
