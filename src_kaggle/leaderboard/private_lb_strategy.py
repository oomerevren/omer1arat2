from __future__ import annotations
from pathlib import Path
import pandas as pd


def _top_candidates(df: pd.DataFrame, status: str, n: int = 10) -> pd.DataFrame:
    part=df[df.get("strategic_status", pd.Series(dtype=str)).eq(status)].copy()
    if part.empty: return part
    return part.sort_values(["OOF macro-F1","class0 F1"], ascending=False).head(n)


def write_private_lb_flags(df: pd.DataFrame, segment: pd.DataFrame, out_dir: str|Path):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lines=["# Private LB Risk Flags", "", "Public LB is treated as the last signal, not a selection oracle.", ""]
    if df.empty:
        lines += ["No experiments available."]
    else:
        for flag in ["PUBLIC_UP_OOF_DOWN","PUBLIC_UP_CLASS0_DOWN","THRESHOLD_FRAGILE","SPLITTER_INCONSISTENT","SEGMENT_COLLAPSE_RISK","ENSEMBLE_OVERFIT_RISK","DENSE_ARTIFACT_RISK","RETRIEVAL_FEATURE_DRIFT","PRIVATE_UNSAFE_CANDIDATE"]:
            count=int(df.get("risk_flag", pd.Series(dtype=str)).astype(str).str.contains(flag, regex=False).sum())
            lines.append(f"- {flag}: {count}")
        if not segment.empty:
            lines.append(f"- SEGMENT_COLLAPSE_RISK: {int(segment.get('segment_collapse_flag', pd.Series(dtype=bool)).sum())}")
    (out/"private_lb_risk_flags.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def write_strategy_recommendation(df: pd.DataFrame, splitter_report: dict, family_report: dict, threshold_report: dict, out_dir: str|Path):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    safe=_top_candidates(df, "private_safe")
    balanced=_top_candidates(df, "balanced_candidate")
    public_opt=_top_candidates(df, "public_optimistic")
    recommended_splitter=splitter_report.get("recommended_splitter", "term_group")
    lines=[
        "# Private Leaderboard Strategy Recommendation", "",
        "## Karar sinyali önceliği", "",
        "1. OOF macro-F1", "2. class 0 F1", "3. splitter reliability", "4. threshold fragility", "5. seed stability", "6. segment collapse risk", "7. model family drift", "8. public LB", "",
        f"## En güvenilir validation splitter", "", f"Öneri: `{recommended_splitter}`", "", splitter_report.get("note", ""), "",
        "## Threshold stratejisi", "", threshold_report.get("recommendation", "Use OOF global best/stable midpoint; avoid public-tuned thresholds."), "",
        "## Model family notu", "", family_report.get("warning", "Use family drift as risk context, not sole ranking."), "",
        "## Safest balanced candidates", "",
    ]
    lines.append(safe[["experiment_id","experiment_name","model_family","OOF macro-F1","class0 F1","public_lb_score","risk_flag"]].to_csv(index=False) if not safe.empty else "No private_safe candidate yet; official OOF/public history is insufficient.")
    lines += ["", "## Balanced candidates", ""]
    lines.append(balanced[["experiment_id","experiment_name","model_family","OOF macro-F1","class0 F1","public_lb_score","risk_flag"]].to_csv(index=False) if not balanced.empty else "No balanced candidate yet.")
    lines += ["", "## Public-optimistic / dikkat", ""]
    lines.append(public_opt[["experiment_id","experiment_name","model_family","OOF macro-F1","class0 F1","public_lb_score","risk_flag"]].to_csv(index=False) if not public_opt.empty else "No public-optimistic candidate detected yet.")
    lines += ["", "## Son hafta protokolü", "", "- Public LB artışı OOF düşüşüyle gelirse final adayından çıkar.", "- Class0 F1 düşerken public artıyorsa private için alarm kabul edilir.", "- Threshold tweak kaynaklı public artışları model iyileşmesi sayılmaz.", "- Dense/retrieval-heavy adaylar fold-safe recheck olmadan ana submission yapılmaz.", "- En az bir balanced/private-safe ve bir class0-heavy savunmacı submission family saklanır."]
    (out/"private_lb_strategy_recommendation.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
