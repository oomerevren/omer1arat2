"""Ablation suite report writers."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src_kaggle.ablation.risk_assessment import assess_risk, keep_risky_drop


def _md_table(df):
    if df is None or len(df) == 0:
        return "_empty_"
    df = df.reset_index() if isinstance(df, pd.Series) else df.copy()
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)

MASTER_COLUMNS = [
    "ablation_id", "experiment_name", "category", "changed_component", "variant_description",
    "splitter", "seed_set", "OOF macro-F1", "class0 F1", "class1 F1", "best_threshold",
    "threshold_fragility", "seed_std", "public_lb_score", "public_oof_delta", "note", "risk_flag",
    "status", "report_dir", "oof_path",
]


def finalize_master(rows: list[dict], baseline_id: str = "feat_all_features") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    baseline = df[df["ablation_id"].eq(baseline_id)]
    baseline_macro = float(baseline["OOF macro-F1"].iloc[0]) if not baseline.empty and pd.notna(baseline["OOF macro-F1"].iloc[0]) else None
    baseline_class0 = float(baseline["class0 F1"].iloc[0]) if not baseline.empty and pd.notna(baseline["class0 F1"].iloc[0]) else None
    df["risk_flag"] = [assess_risk(r, baseline_macro, baseline_class0) for r in df.to_dict("records")]
    for c in MASTER_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[MASTER_COLUMNS]


def write_ablation_reports(master: pd.DataFrame, out_dir: str | Path = "reports/ablation") -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    master.to_csv(out / "ablation_master_table.csv", index=False)
    mapping = {
        "feature": "feature_group_ablation.csv", "negative": "negative_mining_ablation.csv",
        "model": "model_family_ablation.csv", "dense": "dense_ablation.csv", "threshold": "threshold_ablation.csv",
        "retrieval": "retrieval_ablation.csv",
    }
    for cat, fn in mapping.items():
        master[master["category"].eq(cat)].to_csv(out / fn, index=False)
    master[["ablation_id", "category", "changed_component", "risk_flag", "note", "status"]].to_csv(out / "risk_flags.csv", index=False)
    baseline = master[master["ablation_id"].eq("feat_all_features")]
    baseline_macro = float(baseline["OOF macro-F1"].iloc[0]) if not baseline.empty and str(baseline["status"].iloc[0]) == "completed" else None
    krd = master.copy()
    krd["decision"] = [keep_risky_drop(r, baseline_macro) for r in krd.to_dict("records")]
    krd[["ablation_id", "category", "changed_component", "decision", "risk_flag", "OOF macro-F1", "class0 F1", "note"]].to_csv(out / "final_keep_risky_drop.csv", index=False)
    summary = build_summary_markdown(master, krd)
    (out / "ablation_summary.md").write_text(summary, encoding="utf-8")
    (out / "final_pipeline_recommendation.md").write_text(build_recommendation_markdown(master, krd), encoding="utf-8")


def build_summary_markdown(master: pd.DataFrame, krd: pd.DataFrame) -> str:
    lines = ["# Ablation Summary", "", "OOF-first ablation master summary. Official scores are left empty unless explicitly supplied.", ""]
    lines += ["## Status counts", "", _md_table(master.get("status", pd.Series(dtype=str)).value_counts()), ""]
    lines += ["## Risk counts", "", _md_table(master.get("risk_flag", pd.Series(dtype=str)).value_counts()), ""]
    completed = master[master["status"].eq("completed")].copy()
    if not completed.empty:
        lines += ["## Top completed runs by OOF macro-F1", "", _md_table(completed.sort_values("OOF macro-F1", ascending=False).head(20)[["ablation_id", "category", "OOF macro-F1", "class0 F1", "best_threshold", "risk_flag"]]), ""]
    else:
        lines += ["## Official-data note", "", "No official OOF ablation was completed in this workspace. The framework wrote a full executable plan; run `scripts_kaggle/run_ablation_suite.py` after placing official Kaggle files under `data/`.", ""]
    lines += ["## Keep / risky / drop decision counts", "", _md_table(krd["decision"].value_counts()), ""]
    return "\n".join(lines) + "\n"


def build_recommendation_markdown(master: pd.DataFrame, krd: pd.DataFrame) -> str:
    completed = master[master["status"].eq("completed")]
    evidence_note = "OOF evidence available." if not completed.empty else "Official OOF evidence is not available in this workspace; recommendations below are guarded defaults and must be replaced by completed ablation metrics."
    keep = krd[krd["decision"].eq("Keep no matter what")].head(20)
    risky = krd[krd["decision"].eq("Good but risky")].head(20)
    drop = krd[krd["decision"].eq("Drop from final pipeline")].head(20)
    lines = ["# Final Pipeline Recommendation", "", evidence_note, "", "## Tutulacak bileşenler", ""]
    if keep.empty:
        lines += ["- Pair-centric schema/data contract", "- OOF-first validation + threshold tuning", "- Submission safety layer", "- Attribute/gender-age conflict features", "- Negative mining with false-negative safety"]
    else:
        lines.append(_md_table(keep[["ablation_id", "category", "changed_component", "risk_flag"]]))
    lines += ["", "## Opsiyonel ama riskli bileşenler", ""]
    if risky.empty:
        lines += ["- Dense hard negatives: use only with fold-aware rebuild and strict uncertain filtering.", "- Retrieval score/rank features: monitor leakage/artifact risk.", "- Transformer CE: main candidate only after real GPU OOF proves complementary lift.", "- Segment thresholds: analysis only unless stable across seeds/folds."]
    else:
        lines.append(_md_table(risky[["ablation_id", "category", "changed_component", "risk_flag"]]))
    lines += ["", "## Çıkarılacak / sadeleştirilecek bileşenler", ""]
    if drop.empty:
        lines += ["- No component can be dropped without official OOF evidence yet."]
    else:
        lines.append(_md_table(drop[["ablation_id", "category", "changed_component", "risk_flag"]]))
    lines += ["", "## Model family önerisi", "", "Default final candidate: tabular strong baseline + CE OOF if CE adds class-0/class-1 complementary lift. Transformer is helper until real OOF proves it should be core.", "", "## Dense önerisi", "", "Dense should first be used as an auxiliary signal and hard-negative source. It becomes core only if dense_features_plus_dense_negatives beats no_dense_anywhere with stable class0 F1 and no fold-safety alarm.", "", "## Negative mix önerisi", "", "Start with full mix without unsafe dense; add dense hard negatives only with strict false-negative filtering and fold-aware rebuild.", "", "## Threshold önerisi", "", "Use OOF global best or stable-midpoint. Segment thresholds remain analysis-only unless validated across seeds and private-LB simulation."]
    return "\n".join(lines) + "\n"
