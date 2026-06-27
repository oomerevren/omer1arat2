from __future__ import annotations

import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_utils import numeric_tokens, safe_div, tokens
from src_kaggle.features.query_intent import add_query_intent_features, build_query_intent_resources

INTENT_NUMERIC_COLS = [
    "query_length_tokens", "query_char_length", "has_brand_token", "has_category_token",
    "has_color_token", "has_material_token", "has_style_token", "has_gender_token",
    "has_age_token", "has_size_token", "is_short_query", "is_long_query",
    "is_attribute_heavy", "is_brand_heavy", "is_category_heavy", "possible_typo_or_ambiguous",
]


def build_query_features(df: pd.DataFrame, items: pd.DataFrame | None = None) -> pd.DataFrame:
    if set(INTENT_NUMERIC_COLS).issubset(df.columns):
        enriched = df
    else:
        enriched = add_query_intent_features(df, build_query_intent_resources(items) if items is not None else None)
    rows=[]
    for idx, r in enriched.iterrows():
        q = r.get(SCHEMA.query, "")
        qt = tokens(q)
        nums = numeric_tokens(q)
        chars = str(q) if q is not None else ""
        rows.append({
            **{f"query_{c}": r.get(c, 0) for c in INTENT_NUMERIC_COLS},
            "query_numeric_token_count": len(nums),
            "query_unique_token_count": len(set(qt)),
            "query_numeric_density": safe_div(len(nums), len(qt)),
            "query_symbol_density": safe_div(sum(1 for ch in chars if not ch.isalnum() and not ch.isspace()), len(chars)),
        })
    return pd.DataFrame(rows, index=df.index)
