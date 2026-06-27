"""Pair text formatting for semantic pair/cross-encoder models."""
from __future__ import annotations

import pandas as pd
from src_kaggle.data.schema import SCHEMA

FORMATS = {
    "query_title": [SCHEMA.query, SCHEMA.title],
    "query_title_category": [SCHEMA.query, SCHEMA.title, SCHEMA.category],
    "query_title_category_brand": [SCHEMA.query, SCHEMA.title, SCHEMA.category, SCHEMA.brand],
    "full_v1": [SCHEMA.query, SCHEMA.title, SCHEMA.category, SCHEMA.brand, SCHEMA.normalized_attribute_text, SCHEMA.gender, SCHEMA.age_group],
    # Attribute/intent-first variant for ablation. Useful when color/material/gender cues are decisive.
    "full_v2": [SCHEMA.query, SCHEMA.normalized_attribute_text, "query_intent_text", SCHEMA.title, SCHEMA.category, SCHEMA.brand, SCHEMA.gender, SCHEMA.age_group],
}

LABELS = {
    SCHEMA.query: "QUERY",
    SCHEMA.title: "TITLE",
    SCHEMA.category: "CATEGORY",
    SCHEMA.brand: "BRAND",
    SCHEMA.normalized_attribute_text: "ATTR",
    SCHEMA.gender: "GENDER",
    SCHEMA.age_group: "AGE",
    "query_intent_text": "INTENT",
}

INTENT_FIELDS = [
    "detected_brand_candidates",
    "detected_category_candidates",
    "detected_color_candidates",
    "detected_material_candidates",
    "detected_style_candidates",
    "detected_gender_candidates",
    "detected_age_candidates",
]


def _value(row: pd.Series, field: str) -> str:
    if field == "query_intent_text":
        parts = []
        for c in INTENT_FIELDS:
            v = str(row.get(c, "") or "").strip()
            if v:
                parts.append(f"{c.replace('detected_', '').replace('_candidates', '')}: {v}")
        return " ; ".join(parts)
    return str(row.get(field, "") or "").strip()


def build_pair_text(row: pd.Series, format_version: str = "full_v1") -> str:
    fields = FORMATS.get(format_version)
    if fields is None:
        raise ValueError(f"Unknown pair text format: {format_version}. Available={sorted(FORMATS)}")
    parts = []
    for f in fields:
        val = _value(row, f)
        if val:
            parts.append(f"[{LABELS.get(f, f.upper())}] {val}")
    return " [SEP] ".join(parts)


def add_pair_text(df: pd.DataFrame, format_version: str = "full_v1", output_col: str = "pair_text") -> pd.DataFrame:
    out = df.copy()
    out[output_col] = out.apply(lambda r: build_pair_text(r, format_version), axis=1)
    return out


def describe_pair_text_formats() -> dict[str, list[str]]:
    return {k: list(v) for k, v in FORMATS.items()}
