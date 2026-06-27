from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd
from src_kaggle.leaderboard.correlation_analysis import safe_corr, sign_agreement


def splitter_reliability(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows=[]
    for splitter, part in df.groupby(df.get("splitter", pd.Series(dtype=str)).fillna("unknown")):
        corr=safe_corr(part); sign=sign_agreement(part)
        delta=pd.to_numeric(part.get("public_minus_oof"), errors="coerce").dropna()
        rows.append({
            "splitter": splitter or "unknown", "n_experiments": int(len(part)), "n_public": int(corr["n"]),
            "pearson": corr["pearson"], "spearman": corr["spearman"], "sign_agreement_rate": sign.get("sign_agreement_rate"),
            "public_minus_oof_mean": float(delta.mean()) if len(delta) else None,
            "public_minus_oof_std": float(delta.std(ddof=0)) if len(delta) else None,
            "conservative_score": float((-delta.mean()) if len(delta) else 0),
            "high_risk_count": int(part.get("risk_flag", pd.Series(dtype=str)).astype(str).str.contains("PRIVATE_UNSAFE|PUBLIC_UP", regex=True).sum()),
        })
    table=pd.DataFrame(rows)
    if table.empty:
        recommended="term_group"; note="No completed public/OOF data; default to term_group as conservative pair-query leakage guard."
    else:
        scored=table.copy(); scored["score"] = scored["n_public"].fillna(0)*0.1 - scored["high_risk_count"].fillna(0) - scored["public_minus_oof_std"].fillna(0)
        recommended=str(scored.sort_values("score", ascending=False).iloc[0]["splitter"])
        note="Recommendation is statistical only if n_public>=3; otherwise guarded default."
    report={"recommended_splitter": recommended, "note": note, "rows": table.to_dict("records")}
    return table, report


def write_splitter_report(table: pd.DataFrame, report: dict[str, Any], out_dir: str|Path):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"splitter_reliability_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines=["# Splitter Reliability Summary", "", f"Recommended splitter: `{report.get('recommended_splitter')}`", "", report.get("note", ""), ""]
    if not table.empty:
        lines += ["## Table", "", table.to_csv(index=False)]
    (out/"splitter_reliability_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
