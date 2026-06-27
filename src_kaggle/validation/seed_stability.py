from __future__ import annotations
import pandas as pd

def summarize_seed_results(results:list[dict])->dict:
    df=pd.DataFrame(results)
    if df.empty: return {}
    return {
        'seeds': df['seed'].tolist(),
        'macro_f1_mean': float(df['macro_f1'].mean()), 'macro_f1_std': float(df['macro_f1'].std(ddof=0)),
        'class0_f1_mean': float(df['class0_f1'].mean()), 'class0_f1_std': float(df['class0_f1'].std(ddof=0)),
        'class1_f1_mean': float(df['class1_f1'].mean()), 'class1_f1_std': float(df['class1_f1'].std(ddof=0)),
        'threshold_mean': float(df['best_threshold'].mean()), 'threshold_std': float(df['best_threshold'].std(ddof=0)),
    }
