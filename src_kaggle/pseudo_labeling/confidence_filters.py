"""Safety filters for controlled pseudo labeling."""
from __future__ import annotations
import pandas as pd


def agreement_score(row: pd.Series, model_prob_cols: list[str]) -> float:
    if not model_prob_cols:
        return 1.0
    vals=[float(row[c]) for c in model_prob_cols if c in row and pd.notna(row[c])]
    if not vals: return 1.0
    high=sum(v>=0.5 for v in vals); low=sum(v<0.5 for v in vals)
    return max(high,low)/len(vals)


def pseudo_label_decision(row: pd.Series, cfg: dict, model_prob_cols: list[str] | None = None) -> tuple[int|None, str, float]:
    proba=float(row.get('proba', row.get('ensemble_proba', 0.5)) or 0.5)
    model_prob_cols=model_prob_cols or []
    agree=agreement_score(row, model_prob_cols)
    margin=abs(proba-0.5)
    pos_th=float(cfg.get('positive_threshold',0.95)); neg_th=float(cfg.get('negative_threshold',0.05)); min_agree=float(cfg.get('min_agreement',0.66)); min_margin=float(cfg.get('min_margin',0.35))
    mode=cfg.get('mode','disabled')
    if mode=='disabled': return None,'disabled',0.0
    if agree < min_agree or margin < min_margin: return None,'low_agreement_or_margin',agree
    if mode in {'positive_only','dual'} and proba>=pos_th:
        # Require at least some supportive signal unless explicitly disabled.
        if cfg.get('require_support_signals', True):
            support = int(row.get('lex_token_overlap_ratio',0)>0.1 or row.get('retrieval_bm25_score',0)>0 or row.get('sem_query_title_cosine',0)>0.1)
            if not support: return None,'positive_without_support_signals',agree
        return 1,'high_confidence_positive',agree
    if mode in {'negative_only','dual'} and proba<=neg_th:
        conflict = int(row.get('attr_conflict_count',0)>0 or row.get('gender_conflict_flag',0)>0 or row.get('age_conflict_flag',0)>0 or row.get('lex_token_overlap_ratio',1)<0.05)
        if cfg.get('require_negative_conflict', True) and not conflict:
            return None,'negative_without_conflict_signal',agree
        return 0,'high_confidence_negative',agree
    return None,'threshold_not_met',agree
