from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def model_family_drift(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows=[]
    for fam, part in df.groupby(df.get("model_family", pd.Series(dtype=str)).fillna("unknown")):
        delta=pd.to_numeric(part.get("public_minus_oof"), errors="coerce").dropna()
        oof=pd.to_numeric(part.get("OOF macro-F1"), errors="coerce").dropna()
        c0=pd.to_numeric(part.get("class0 F1"), errors="coerce").dropna()
        risk=part.get("risk_flag", pd.Series(dtype=str)).astype(str)
        rows.append({
            "model_family": fam or "unknown", "n_experiments": int(len(part)), "n_public": int(delta.notna().sum()),
            "oof_mean": float(oof.mean()) if len(oof) else None, "class0_mean": float(c0.mean()) if len(c0) else None,
            "public_minus_oof_mean": float(delta.mean()) if len(delta) else None, "public_minus_oof_std": float(delta.std(ddof=0)) if len(delta) else None,
            "public_optimistic_count": int(risk.str.contains("PUBLIC_UP|PRIVATE_UNSAFE", regex=True).sum()),
            "dense_or_retrieval_risk_count": int(risk.str.contains("DENSE_ARTIFACT|RETRIEVAL_FEATURE", regex=True).sum()),
        })
    table=pd.DataFrame(rows)
    report={"warning":"Do not rank families from public LB unless n_public is sufficient; use OOF/class0/seed stability first.", "families": table.to_dict("records")}
    return table, report


def write_model_family_report(table: pd.DataFrame, report: dict, out_dir: str|Path):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out/"model_family_comparison.csv", index=False)
    (out/"model_family_drift_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines=["# Model Family Drift Summary", "", report.get("warning", ""), ""]
    if not table.empty: lines += ["## Family comparison CSV preview", "", table.to_csv(index=False)]
    (out/"model_family_drift_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
