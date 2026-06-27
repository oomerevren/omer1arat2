"""OOF model comparison utilities."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src_kaggle.data.schema import SCHEMA
from src_kaggle.validation.threshold_optimizer import optimize_oof_thresholds
from src_kaggle.validation.segment_reports import segment_scores


def _load(path):
    return pd.read_csv(path)


def compare_oof_models(oof_paths: dict[str, str], out_dir: str | Path = "reports/experiments/comparison") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    frames = {name: _load(path) for name, path in oof_paths.items()}
    rows = []
    for name, df in frames.items():
        opt = optimize_oof_thresholds(df, segment_min_rows=10)
        rows.append({
            "model": name, "macro_f1": opt.best_metrics["macro_f1"], "class0_f1": opt.best_metrics["class0_f1"],
            "class1_f1": opt.best_metrics["class1_f1"], "best_threshold": opt.best_threshold,
            "positive_prediction_rate": opt.best_metrics["positive_prediction_rate"],
        })
        seg = segment_scores(df, opt.best_threshold, min_rows=10)
        seg["model"] = name
        seg.to_csv(out / f"{name}_segment_scores.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    summary.to_csv(out / "model_summary.csv", index=False)

    names = list(frames)
    corr = pd.DataFrame(index=names, columns=names, dtype=float)
    disagreement_rows = []
    for a in names:
        for b in names:
            merged = frames[a][["id", "label", "proba"]].merge(frames[b][["id", "proba"]], on="id", suffixes=(f"_{a}", f"_{b}"))
            corr.loc[a, b] = float(np.corrcoef(merged[f"proba_{a}"], merged[f"proba_{b}"])[0, 1]) if len(merged) > 1 else 1.0
            if a < b:
                pa = (merged[f"proba_{a}"] >= 0.5).astype(int); pb = (merged[f"proba_{b}"] >= 0.5).astype(int)
                diff = merged[pa != pb].copy(); diff["model_a"] = a; diff["model_b"] = b
                disagreement_rows.append(diff.head(200))
    corr.to_csv(out / "prediction_correlation.csv")
    if disagreement_rows:
        pd.concat(disagreement_rows, ignore_index=True).to_csv(out / "model_disagreements.csv", index=False)
    result = {"models": rows, "correlation_path": str(out / "prediction_correlation.csv"), "summary_path": str(out / "model_summary.csv")}
    (out / "comparison_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return result
