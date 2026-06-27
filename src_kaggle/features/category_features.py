from __future__ import annotations

import re
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_utils import safe_div, token_set, tokens, normalize_text

SEP_RE = re.compile(r"\s*(?:>|/|\\|\||;|,|-)\s*")


def _category_parts(cat) -> list[str]:
    cat = normalize_text(cat)
    if not cat:
        return []
    return [p for p in SEP_RE.split(cat) if p]


def build_category_features(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        q = r.get(SCHEMA.query, ""); cat = r.get(SCHEMA.category, "")
        parts = _category_parts(cat)
        parent = parts[0] if parts else ""; leaf = parts[-1] if parts else ""
        qset, cset = token_set(q), token_set(cat)
        pset, lset = token_set(parent), token_set(leaf)
        inter = qset & cset
        rows.append({
            "cat_depth": len(parts),
            "cat_exact_full_match_flag": int(bool(normalize_text(cat)) and normalize_text(cat) in normalize_text(q)),
            "cat_parent_overlap_flag": int(bool(qset & pset)),
            "cat_leaf_overlap_flag": int(bool(qset & lset)),
            "cat_token_overlap_count": len(inter),
            "cat_token_overlap_ratio": safe_div(len(inter), len(qset | cset)),
            "cat_query_coverage_ratio": safe_div(len(inter), len(qset)),
            "cat_category_coverage_ratio": safe_div(len(inter), len(cset)),
            "cat_leaf_query_coverage_ratio": safe_div(len(qset & lset), len(lset)),
        })
    return pd.DataFrame(rows, index=df.index)
