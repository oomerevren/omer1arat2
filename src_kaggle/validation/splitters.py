from __future__ import annotations
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_utils import ascii_fold

class TermGroupSplitter:
    def __init__(self,n_splits=5,seed=42): self.n_splits=n_splits; self.seed=seed
    def split(self,df):
        y=df[SCHEMA.label].astype(int); groups=df[SCHEMA.term_id]
        if groups.nunique()>=self.n_splits:
            yield from GroupKFold(self.n_splits).split(df,y,groups)
        else:
            yield from StratifiedKFold(self.n_splits,shuffle=True,random_state=self.seed).split(df,y)

class QueryGroupSplitter:
    def __init__(self,n_splits=5,seed=42): self.n_splits=n_splits; self.seed=seed
    def split(self,df):
        y=df[SCHEMA.label].astype(int); groups=df[SCHEMA.query].map(ascii_fold)
        if groups.nunique()>=self.n_splits:
            yield from GroupKFold(self.n_splits).split(df,y,groups)
        else:
            yield from StratifiedKFold(self.n_splits,shuffle=True,random_state=self.seed).split(df,y)

class TailAwareSplitter:
    """Group split with deterministic tail-query ordering hook; currently GroupKFold-safe."""
    def __init__(self,n_splits=5,seed=42,group_col=SCHEMA.term_id): self.n_splits=n_splits; self.seed=seed; self.group_col=group_col
    def split(self,df):
        # Keeps leakage safety first; tail weighting can be enhanced after real frequency analysis.
        groups=df[self.group_col] if self.group_col in df.columns else df[SCHEMA.term_id]
        y=df[SCHEMA.label].astype(int)
        if groups.nunique()>=self.n_splits: yield from GroupKFold(self.n_splits).split(df,y,groups)
        else: yield from StratifiedKFold(self.n_splits,shuffle=True,random_state=self.seed).split(df,y)

def get_splitter(name:str,n_splits:int,seed:int):
    if name=='query_group': return QueryGroupSplitter(n_splits,seed)
    if name=='tail_aware': return TailAwareSplitter(n_splits,seed)
    return TermGroupSplitter(n_splits,seed)
