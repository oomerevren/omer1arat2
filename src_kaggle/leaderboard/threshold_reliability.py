from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def threshold_reliability(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out=df[[c for c in ["experiment_id","experiment_name","model_family","threshold","OOF macro-F1","class0 F1","class1 F1","public_lb_score","public_minus_oof","threshold_fragility","seed_std","risk_flag"] if c in df.columns]].copy()
    out["threshold"] = pd.to_numeric(out.get("threshold"), errors="coerce")
    out["threshold_extreme_flag"] = out["threshold"].map(lambda x: bool(pd.notna(x) and (x < 0.20 or x > 0.80)))
    out["threshold_public_risk_flag"] = out.apply(lambda r: bool(str(r.get("threshold_fragility", "")).lower() in {"true","1"} and pd.notna(r.get("public_lb_score"))), axis=1)
    report={
        "n_threshold_rows": int(len(out)),
        "fragile_count": int(out.get("threshold_fragility", pd.Series(dtype=str)).astype(str).str.lower().isin(["true","1"]).sum()),
        "extreme_threshold_count": int(out["threshold_extreme_flag"].sum()) if "threshold_extreme_flag" in out else 0,
        "recommendation": "Prefer OOF global best or stable midpoint. Do not tune threshold directly to public LB; segment thresholds are analysis-only unless stable across seeds.",
    }
    return out, report


def write_threshold_report(table: pd.DataFrame, report: dict, out_dir: str|Path):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out/"threshold_public_drift.csv", index=False)
    (out/"threshold_reliability_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
