from __future__ import annotations

import pandas as pd

from src_kaggle.data.attribute_parser import parse_attribute_dict_json
from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_utils import numeric_tokens, safe_div, token_set


def _vals(d: dict, key: str) -> set[str]:
    return set(map(str, d.get(key, [])))


def build_attribute_features(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        ad = parse_attribute_dict_json(r.get(SCHEMA.attribute_dict, ""))
        keys = set(ad.keys())
        vals = set(v for vs in ad.values() for v in vs)
        qset = token_set(r.get(SCHEMA.query, ""))
        attr_text_set = token_set(r.get(SCHEMA.normalized_attribute_text, r.get(SCHEMA.attributes, "")))
        q_colors = set(filter(None, str(r.get("detected_color_candidates", "")).split("|")))
        q_mats = set(filter(None, str(r.get("detected_material_candidates", "")).split("|")))
        q_styles = set(filter(None, str(r.get("detected_style_candidates", "")).split("|")))
        item_colors, item_mats, item_styles = _vals(ad,"color"), _vals(ad,"material"), _vals(ad,"style")
        qnums = numeric_tokens(r.get(SCHEMA.query,"")); attrnums = numeric_tokens(r.get(SCHEMA.attributes,"")) | numeric_tokens(r.get(SCHEMA.title,""))
        color_match = bool(q_colors and item_colors and not q_colors.isdisjoint(item_colors))
        material_match = bool(q_mats and item_mats and not q_mats.isdisjoint(item_mats))
        style_match = bool(q_styles and item_styles and not q_styles.isdisjoint(item_styles))
        conflicts = int(bool(q_colors and item_colors and q_colors.isdisjoint(item_colors))) + int(bool(q_mats and item_mats and q_mats.isdisjoint(item_mats))) + int(bool(q_styles and item_styles and q_styles.isdisjoint(item_styles)))
        rows.append({
            "attr_key_count": len(keys),
            "attr_value_count": len(vals),
            "attr_key_overlap_count": len(qset & keys),
            "attr_value_overlap_count": len(qset & vals),
            "attr_text_token_overlap_count": len(qset & attr_text_set),
            "attr_coverage_ratio": safe_div(len(qset & attr_text_set), len(qset)),
            "attr_color_exact_match": int(color_match),
            "attr_color_conflict_flag": int(bool(q_colors and item_colors and q_colors.isdisjoint(item_colors))),
            "attr_color_missing_when_query_has_color": int(bool(q_colors and not item_colors)),
            "attr_material_exact_match": int(material_match),
            "attr_material_conflict_flag": int(bool(q_mats and item_mats and q_mats.isdisjoint(item_mats))),
            "attr_material_missing_when_query_has_material": int(bool(q_mats and not item_mats)),
            "attr_style_overlap": int(style_match),
            "attr_size_numeric_match": int(bool(qnums & attrnums)),
            "attr_query_has_numeric_no_item_numeric": int(bool(qnums and not attrnums)),
            "attr_conflict_count": conflicts,
        })
    return pd.DataFrame(rows, index=df.index)
