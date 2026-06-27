from __future__ import annotations

import pandas as pd

from src_kaggle.data.attribute_parser import parse_attribute_dict_json
from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_utils import is_unknown, numeric_tokens, safe_div, tokens, normalize_text


def build_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        title_t = tokens(r.get(SCHEMA.title,"")); query_t = tokens(r.get(SCHEMA.query,""))
        fields = [SCHEMA.query, SCHEMA.title, SCHEMA.category, SCHEMA.brand, SCHEMA.gender, SCHEMA.age_group, SCHEMA.attributes]
        missing = sum(1 for f in fields if normalize_text(r.get(f,"")) == "")
        unknown = sum(1 for f in [SCHEMA.gender, SCHEMA.age_group] if is_unknown(r.get(f,"")))
        ad = parse_attribute_dict_json(r.get(SCHEMA.attribute_dict,""))
        chars = str(r.get(SCHEMA.title,"") or "")
        rows.append({
            "meta_title_token_count": len(title_t),
            "meta_query_title_len_ratio": safe_div(len(query_t), len(title_t)),
            "meta_brand_present_flag": int(not is_unknown(r.get(SCHEMA.brand,""))),
            "meta_attribute_count": sum(len(v) for v in ad.values()),
            "meta_missing_field_count": missing,
            "meta_unknown_field_count": unknown,
            "meta_title_numeric_density": safe_div(len(numeric_tokens(r.get(SCHEMA.title,""))), len(title_t)),
            "meta_title_symbol_density": safe_div(sum(1 for ch in chars if not ch.isalnum() and not ch.isspace()), len(chars)),
        })
    return pd.DataFrame(rows, index=df.index)
