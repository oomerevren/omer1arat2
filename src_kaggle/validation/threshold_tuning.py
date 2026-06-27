from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    y=np.asarray(y_true).astype(int); p=(np.asarray(proba)>=threshold).astype(int)
    macro=f1_score(y,p,average='macro',zero_division=0)
    pr,rc,f1,_=precision_recall_fscore_support(y,p,labels=[0,1],zero_division=0)
    cm=confusion_matrix(y,p,labels=[0,1])
    return {
        'threshold': float(threshold), 'macro_f1': float(macro),
        'class0_precision': float(pr[0]), 'class0_recall': float(rc[0]), 'class0_f1': float(f1[0]),
        'class1_precision': float(pr[1]), 'class1_recall': float(rc[1]), 'class1_f1': float(f1[1]),
        'positive_prediction_rate': float(p.mean()) if len(p) else 0.0,
        'tn': int(cm[0,0]), 'fp': int(cm[0,1]), 'fn': int(cm[1,0]), 'tp': int(cm[1,1]),
    }


def threshold_curve(y_true, proba, start=0.05, end=0.95, step=0.01) -> pd.DataFrame:
    thresholds=np.round(np.arange(start, end+1e-9, step), 10)
    return pd.DataFrame([metrics_at_threshold(y_true, proba, float(t)) for t in thresholds])


def find_best_threshold(y_true, proba, start=0.05, end=0.95, step=0.01) -> tuple[float, dict, pd.DataFrame]:
    curve=threshold_curve(y_true, proba, start, end, step)
    idx=curve['macro_f1'].idxmax()
    row=curve.loc[idx].to_dict()
    return float(row['threshold']), row, curve
