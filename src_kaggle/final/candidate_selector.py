"""Final candidate pool selection from OOF/public/private-risk intelligence."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def select_final_candidates(
    leaderboard_path: str | Path = "reports/leaderboard/oof_public_correlation.csv",
    ablation_path: str | Path = "reports/ablation/ablation_master_table.csv",
    max_candidates: int = 20,
) -> pd.DataFrame:
    """Create final candidate pool.

    If real OOF rows do not exist yet, this still writes a transparent candidate
    planning table with `selection_status=not_ready_no_oof` rather than inventing
    scores.
    """
    lb = _read(leaderboard_path)
    if lb.empty:
        lb = pd.DataFrame(columns=["experiment_id","experiment_name","model_family","splitter","threshold","OOF macro-F1","class0 F1","class1 F1","seed_std","public_lb_score","risk_flag","strategic_status","oof_path","notes"])
    rows: list[dict[str, Any]] = []
    for _, r in lb.iterrows():
        oof = pd.to_numeric(pd.Series([r.get("OOF macro-F1")]), errors="coerce").iloc[0]
        c0 = pd.to_numeric(pd.Series([r.get("class0 F1")]), errors="coerce").iloc[0]
        status = "ready" if pd.notna(oof) else "not_ready_no_oof"
        risk = str(r.get("risk_flag", ""))
        strategic = str(r.get("strategic_status", ""))
        # candidate score: OOF first, class0 second, public last/weak.
        pub = pd.to_numeric(pd.Series([r.get("public_lb_score")]), errors="coerce").iloc[0]
        score = (float(oof) if pd.notna(oof) else -1.0) + 0.15 * (float(c0) if pd.notna(c0) else 0.0) + 0.02 * (float(pub) if pd.notna(pub) else 0.0)
        if "PUBLIC_UP_OOF_DOWN" in risk or "PUBLIC_UP_CLASS0_DOWN" in risk or "PRIVATE_UNSAFE" in risk:
            score -= 0.10
        rows.append({
            "candidate_id": r.get("experiment_id") or r.get("experiment_name"),
            "experiment_id": r.get("experiment_id", ""),
            "experiment_name": r.get("experiment_name", ""),
            "model_family": r.get("model_family", "unknown"),
            "data_feature_negative_versions": r.get("notes", ""),
            "OOF macro-F1": oof,
            "class0 F1": c0,
            "class1 F1": pd.to_numeric(pd.Series([r.get("class1 F1")]), errors="coerce").iloc[0],
            "threshold": pd.to_numeric(pd.Series([r.get("threshold")]), errors="coerce").iloc[0],
            "seed_std": pd.to_numeric(pd.Series([r.get("seed_std")]), errors="coerce").iloc[0],
            "public_lb_score": pub,
            "risk_label": risk,
            "strategic_status": strategic,
            "splitter": r.get("splitter", ""),
            "oof_path": r.get("oof_path", ""),
            "test_pred_path": r.get("test_pred_path", ""),
            "selection_score": score,
            "selection_status": status,
        })
    pool = pd.DataFrame(rows)
    if pool.empty or pool["selection_status"].eq("ready").sum() == 0:
        # Create explicit planned family placeholders from ablation table so reports
        # remain useful before official data arrives.
        planned = [
            ("planned_tabular_balanced", "tabular", "Balanced tabular OOF candidate required"),
            ("planned_class0_defensive", "tabular", "Class0/precision defensive candidate required"),
            ("planned_semantic_aggressive", "dense_enhanced", "Dense/CE semantic challenger candidate required"),
        ]
        pool = pd.DataFrame([{
            "candidate_id": cid, "experiment_id": "", "experiment_name": cid, "model_family": fam,
            "data_feature_negative_versions": note, "OOF macro-F1": pd.NA, "class0 F1": pd.NA, "class1 F1": pd.NA,
            "threshold": pd.NA, "seed_std": pd.NA, "public_lb_score": pd.NA, "risk_label": "needs_real_oof",
            "strategic_status": "high_risk_experimental" if "semantic" in cid else "not_ready",
            "splitter": "term_group", "oof_path": "", "test_pred_path": "", "selection_score": -1.0,
            "selection_status": "not_ready_no_oof",
        } for cid, fam, note in planned])
    else:
        ready = pool[pool["selection_status"].eq("ready")].sort_values("selection_score", ascending=False).head(max_candidates)
        # keep some not-ready rows for audit only if no ready? no.
        pool = ready.reset_index(drop=True)
    return pool


def write_candidate_pool(pool: pd.DataFrame, path: str | Path = "reports/final/final_candidate_pool.csv") -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(p, index=False)
