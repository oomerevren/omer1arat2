from __future__ import annotations
import pandas as pd
from src_kaggle.data.schema import SCHEMA
from src_kaggle.validation.threshold_tuning import metrics_at_threshold

DEFAULT_SEGMENTS=['is_short_query','is_long_query','is_brand_heavy','is_attribute_heavy','is_category_heavy','has_gender_token','has_age_token','negative_type','source_pool',SCHEMA.category,SCHEMA.gender,SCHEMA.age_group]

def segment_scores(oof:pd.DataFrame, threshold:float, segments=None, min_rows:int=5)->pd.DataFrame:
    segments=segments or DEFAULT_SEGMENTS; rows=[]
    for seg in segments:
        if seg not in oof.columns: continue
        for val, part in oof.groupby(seg, dropna=False):
            if len(part)<min_rows: continue
            m=metrics_at_threshold(part[SCHEMA.label], part['proba'], threshold)
            m.update({'segment':seg,'value':str(val),'n':int(len(part))}); rows.append(m)
    return pd.DataFrame(rows)
