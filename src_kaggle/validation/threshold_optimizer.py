"""Professional OOF threshold optimization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.validation.threshold_tuning import threshold_curve, metrics_at_threshold, find_best_threshold

SEGMENT_COLUMNS = [
    "is_short_query", "is_brand_heavy", "is_attribute_heavy", "is_category_heavy",
    "negative_type", "source_pool", "has_gender_token", "has_age_token",
]

@dataclass
class ThresholdOptimizationResult:
    best_threshold: float
    best_metrics: dict
    curve: pd.DataFrame
    fold_thresholds: pd.DataFrame
    segment_thresholds: pd.DataFrame
    sensitivity: dict


def _threshold_sensitivity(curve: pd.DataFrame, best_threshold: float, eps: float = 0.03) -> dict:
    best = float(curve["macro_f1"].max()) if not curve.empty else 0.0
    near = curve[(curve["threshold"] >= best_threshold - eps) & (curve["threshold"] <= best_threshold + eps)]
    if near.empty:
        return {"window": eps, "macro_f1_drop_max": 0.0, "is_fragile": False}
    drop = float(best - near["macro_f1"].min())
    return {
        "window": float(eps),
        "macro_f1_best": best,
        "macro_f1_min_in_window": float(near["macro_f1"].min()),
        "macro_f1_drop_max": drop,
        "is_fragile": bool(drop > 0.01),
    }


def optimize_oof_thresholds(
    oof: pd.DataFrame,
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
    segment_min_rows: int = 30,
) -> ThresholdOptimizationResult:
    best_t, best_metrics, curve = find_best_threshold(oof[SCHEMA.label], oof["proba"], start, end, step)
    fold_rows = []
    if "fold" in oof.columns:
        for fold, part in oof.groupby("fold"):
            t, m, _ = find_best_threshold(part[SCHEMA.label], part["proba"], start, end, step)
            m.update({"fold": fold, "best_threshold": t, "n": len(part)})
            fold_rows.append(m)
    seg_rows = []
    for seg in SEGMENT_COLUMNS:
        if seg not in oof.columns:
            continue
        for val, part in oof.groupby(seg, dropna=False):
            if len(part) < segment_min_rows or part[SCHEMA.label].nunique() < 2:
                continue
            t, m, _ = find_best_threshold(part[SCHEMA.label], part["proba"], start, end, step)
            global_m = metrics_at_threshold(part[SCHEMA.label], part["proba"], best_t)
            m.update({
                "segment": seg, "value": str(val), "n": len(part), "best_threshold": t,
                "global_threshold_macro_f1": global_m["macro_f1"],
                "segment_gain_vs_global": float(m["macro_f1"] - global_m["macro_f1"]),
                "overfit_risk_note": "high" if len(part) < 100 else "medium" if len(part) < 500 else "lower",
            })
            seg_rows.append(m)
    return ThresholdOptimizationResult(
        best_threshold=best_t,
        best_metrics=best_metrics,
        curve=curve,
        fold_thresholds=pd.DataFrame(fold_rows),
        segment_thresholds=pd.DataFrame(seg_rows),
        sensitivity=_threshold_sensitivity(curve, best_t),
    )


def write_threshold_report(result: ThresholdOptimizationResult, out_dir: str | Path) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    result.curve.to_csv(out / "threshold_curve.csv", index=False)
    result.fold_thresholds.to_csv(out / "fold_thresholds.csv", index=False)
    result.segment_thresholds.to_csv(out / "segment_thresholds.csv", index=False)
    summary = {
        "best_threshold": result.best_threshold,
        "best_metrics": result.best_metrics,
        "sensitivity": result.sensitivity,
        "fold_threshold_std": float(result.fold_thresholds["best_threshold"].std(ddof=0)) if not result.fold_thresholds.empty else 0.0,
        "segment_threshold_count": int(len(result.segment_thresholds)),
    }
    (out / "threshold_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary
