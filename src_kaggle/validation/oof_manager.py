from __future__ import annotations
from pathlib import Path
import pandas as pd
from src_kaggle.data.schema import SCHEMA

def make_oof(df:pd.DataFrame, proba, folds, threshold:float, model_name:str)->pd.DataFrame:
    cols=[c for c in [SCHEMA.id,SCHEMA.term_id,SCHEMA.item_id,SCHEMA.label,SCHEMA.query,'negative_type','source_pool','is_short_query','is_long_query','is_brand_heavy','is_attribute_heavy','is_category_heavy','has_gender_token','has_age_token',SCHEMA.category,SCHEMA.gender,SCHEMA.age_group] if c in df.columns]
    out=df[cols].copy(); out['fold']=folds; out['model_name']=model_name; out['proba']=proba
    out['pred_default']=(out['proba']>=0.5).astype(int); out['pred_best_threshold']=(out['proba']>=threshold).astype(int)
    return out

def write_oof(oof:pd.DataFrame,path:str|Path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); oof.to_csv(path,index=False)
