from __future__ import annotations

import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_utils import normalize_text, safe_div, token_set, tokens


def build_brand_features(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        q = r.get(SCHEMA.query, ""); brand = r.get(SCHEMA.brand, "")
        qn, bn = normalize_text(q), normalize_text(brand)
        qset, bset = token_set(q), token_set(brand)
        detected = str(r.get("detected_brand_candidates", "") or "")
        detected_set = set(filter(None, detected.split("|")))
        brand_in_query = bool(bn and (bn in qn or bool(qset & bset)))
        has_brand_intent = bool(int(r.get("has_brand_token", 0)) or detected_set or brand_in_query)
        exact = bool(bn and bn in qn)
        contradiction = has_brand_intent and not brand_in_query
        rows.append({
            "brand_present_flag": int(bool(bn)),
            "brand_query_has_brand_flag": int(has_brand_intent),
            "brand_exact_match": int(exact),
            "brand_token_overlap_count": len(qset & bset),
            "brand_token_overlap_ratio": safe_div(len(qset & bset), len(bset)),
            "brand_contradiction_flag": int(contradiction),
            "brand_token_length": len(tokens(brand)),
            "brand_only_query_flag": int(bool(qset) and bool(bset) and qset.issubset(bset | detected_set)),
            "brand_detected_candidate_count": len(detected_set),
        })
    return pd.DataFrame(rows, index=df.index)
