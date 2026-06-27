from __future__ import annotations

import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_utils import normalize_text, is_unknown


def build_gender_age_features(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        gender = normalize_text(r.get(SCHEMA.gender,"")); age = normalize_text(r.get(SCHEMA.age_group,""))
        qg = set(filter(None, str(r.get("detected_gender_candidates", "")).split("|")))
        qa = set(filter(None, str(r.get("detected_age_candidates", "")).split("|")))
        female_item = "kad" in gender or "female" in gender
        male_item = "erkek" in gender or gender == "male"
        unisex_item = "unisex" in gender
        baby_item = "bebek" in age or "baby" in age
        child_item = any(x in age for x in ["çocuk","cocuk","child","bebek"])
        gender_match = ("female" in qg and female_item) or ("male" in qg and male_item) or ("unisex" in qg and unisex_item)
        gender_conflict = (("female" in qg and male_item) or ("male" in qg and female_item)) and not unisex_item
        age_match = ("baby" in qa and baby_item) or ("child" in qa and child_item) or ("school_age" in qa and child_item)
        age_conflict = (("baby" in qa and not baby_item) or ("child" in qa and not child_item)) if qa else False
        rows.append({
            "gender_query_has_cue": int(bool(qg)),
            "age_query_has_cue": int(bool(qa)),
            "gender_item_unknown_flag": int(is_unknown(gender)),
            "age_item_unknown_flag": int(is_unknown(age)),
            "gender_exact_match_flag": int(gender_match),
            "gender_conflict_flag": int(gender_conflict),
            "age_exact_match_flag": int(age_match),
            "age_conflict_flag": int(age_conflict),
            "gender_item_unisex_flag": int(unisex_item),
        })
    return pd.DataFrame(rows, index=df.index)
