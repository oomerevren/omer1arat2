"""Link OOF/validation experiments, submissions and manual public-LB entries.

This module never treats public LB as ground truth.  It only creates a traceable
join table so private-LB defense reports can reason about OOF/public divergence.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _coalesce(*vals):
    for v in vals:
        if pd.notna(v) and str(v) != "":
            return v
    return ""


def infer_model_family(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(k, "")) for k in ["model_type", "experiment_name", "models", "category", "changed_component", "note", "submission_note"]).lower()
    if "ensemble" in text or "blend" in text or "+" in text:
        return "ensemble"
    if "transformer" in text or "ce_best" in text:
        return "transformer_ce"
    if "sklearn_text" in text or "cross_encoder" in text or "pair-text" in text:
        return "sklearn_text"
    if "dense" in text:
        return "dense_enhanced"
    if "retrieval" in text:
        return "retrieval_heavy"
    if "tab" in text or "tabular" in text or str(row.get("model_type", "")).lower() == "tabular":
        return "tabular"
    return "unknown"


def normalize_experiment_registry(path: str | Path = "reports/experiments/experiment_registry.csv") -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["experiment_id", "experiment_name", "model_family", "splitter", "threshold", "OOF macro-F1", "class0 F1", "class1 F1", "threshold_fragility", "seed_std", "report_dir", "oof_path", "public_lb_score", "notes"])
    out = pd.DataFrame()
    out["experiment_id"] = df.get("experiment_id", "")
    out["experiment_name"] = df.get("experiment_name", "")
    out["model_family"] = [infer_model_family(r) for r in df.to_dict("records")]
    out["splitter"] = df.get("validation_version", "")
    out["threshold"] = pd.to_numeric(df.get("best_threshold", pd.Series(dtype=float)), errors="coerce")
    out["OOF macro-F1"] = pd.to_numeric(df.get("oof_macro_f1", pd.Series(dtype=float)), errors="coerce")
    out["class0 F1"] = pd.to_numeric(df.get("class0_f1", pd.Series(dtype=float)), errors="coerce")
    out["class1 F1"] = pd.to_numeric(df.get("class1_f1", pd.Series(dtype=float)), errors="coerce")
    out["threshold_fragility"] = df.get("threshold_fragile", "")
    out["seed_std"] = pd.NA
    out["report_dir"] = df.get("report_dir", "")
    out["oof_path"] = df.get("oof_path", "")
    out["public_lb_score"] = pd.to_numeric(df.get("public_lb_score", pd.Series(dtype=float)), errors="coerce")
    out["notes"] = df.get("submission_note", "")
    out["source_registry"] = "experiment_registry"
    return out


def normalize_ablation_master(path: str | Path = "reports/ablation/ablation_master_table.csv") -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["experiment_id"] = df.get("ablation_id", "")
    out["experiment_name"] = df.get("experiment_name", "")
    out["model_family"] = [infer_model_family(r) for r in df.to_dict("records")]
    out["splitter"] = df.get("splitter", "")
    out["threshold"] = pd.to_numeric(df.get("best_threshold", pd.Series(dtype=float)), errors="coerce")
    out["OOF macro-F1"] = pd.to_numeric(df.get("OOF macro-F1", pd.Series(dtype=float)), errors="coerce")
    out["class0 F1"] = pd.to_numeric(df.get("class0 F1", pd.Series(dtype=float)), errors="coerce")
    out["class1 F1"] = pd.to_numeric(df.get("class1 F1", pd.Series(dtype=float)), errors="coerce")
    out["threshold_fragility"] = df.get("threshold_fragility", "")
    out["seed_std"] = pd.to_numeric(df.get("seed_std", pd.Series(dtype=float)), errors="coerce")
    out["report_dir"] = df.get("report_dir", "")
    out["oof_path"] = df.get("oof_path", "")
    out["public_lb_score"] = pd.to_numeric(df.get("public_lb_score", pd.Series(dtype=float)), errors="coerce")
    out["notes"] = df.get("note", "")
    out["source_registry"] = "ablation_master"
    out["ablation_category"] = df.get("category", "")
    out["changed_component"] = df.get("changed_component", "")
    out["status"] = df.get("status", "")
    return out


def normalize_submission_registry(path: str | Path = "reports/submissions/submission_registry.csv") -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["submission_id", "experiment_id", "file_path", "submission_timestamp", "public_lb_score", "chosen_family_label", "submission_notes", "validation_score", "threshold"])
    out = pd.DataFrame()
    out["submission_id"] = [f"sub_{i:04d}" for i in range(len(df))]
    out["experiment_id"] = df.get("experiment_id", "")
    out["file_path"] = df.get("file_path", "")
    out["submission_timestamp"] = df.get("timestamp", "")
    out["public_lb_score"] = pd.to_numeric(df.get("public_lb_score", pd.Series(dtype=float)), errors="coerce")
    out["chosen_family_label"] = df.get("chosen_family_label", df.get("models", ""))
    out["submission_notes"] = df.get("note", "")
    out["validation_score"] = pd.to_numeric(df.get("validation_score", pd.Series(dtype=float)), errors="coerce")
    out["threshold"] = pd.to_numeric(df.get("threshold", pd.Series(dtype=float)), errors="coerce")
    return out


def load_public_lb_entries(path: str | Path = "reports/leaderboard/public_lb_tracking_table.csv") -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["submission_id", "experiment_id", "experiment_name", "file_path", "public_lb_score", "submission_timestamp", "chosen_family_label", "notes"])
    df["public_lb_score"] = pd.to_numeric(df.get("public_lb_score", pd.Series(dtype=float)), errors="coerce")
    return df


def append_public_lb_entry(
    *,
    file_path: str = "",
    public_lb_score: float | str = "",
    experiment_id: str = "",
    experiment_name: str = "",
    submission_timestamp: str | None = None,
    notes: str = "",
    chosen_family_label: str = "",
    tracking_path: str | Path = "reports/leaderboard/public_lb_tracking_table.csv",
) -> pd.DataFrame:
    path = Path(tracking_path); path.parent.mkdir(parents=True, exist_ok=True)
    old = load_public_lb_entries(path)
    row = {
        "submission_id": f"manual_{len(old):04d}",
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "file_path": file_path,
        "public_lb_score": public_lb_score,
        "submission_timestamp": submission_timestamp or datetime.now(timezone.utc).isoformat(),
        "chosen_family_label": chosen_family_label,
        "notes": notes,
    }
    df = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def build_oof_public_table(
    *,
    experiment_registry: str | Path = "reports/experiments/experiment_registry.csv",
    submission_registry: str | Path = "reports/submissions/submission_registry.csv",
    ablation_master: str | Path = "reports/ablation/ablation_master_table.csv",
    public_lb_tracking: str | Path = "reports/leaderboard/public_lb_tracking_table.csv",
) -> pd.DataFrame:
    exp = normalize_experiment_registry(experiment_registry)
    abl = normalize_ablation_master(ablation_master)
    base = pd.concat([x for x in [exp, abl] if x is not None and not x.empty], ignore_index=True, sort=False)
    if base.empty:
        base = pd.DataFrame(columns=["experiment_id", "experiment_name", "model_family", "splitter", "threshold", "OOF macro-F1", "class0 F1", "class1 F1"])
    subs = normalize_submission_registry(submission_registry)
    manual = load_public_lb_entries(public_lb_tracking)

    # Submission registry join by experiment_id; manual entries can join by id/name/file.
    table = base.copy()
    if not subs.empty:
        sub_keep = subs.sort_values("submission_timestamp").drop_duplicates("experiment_id", keep="last")
        table = table.merge(sub_keep, on="experiment_id", how="left", suffixes=("", "_submission"))
    if not manual.empty:
        man = manual.sort_values("submission_timestamp").drop_duplicates("experiment_id", keep="last") if "experiment_id" in manual else manual
        table = table.merge(man, on="experiment_id", how="left", suffixes=("", "_manual"))

    # Coalesce public score from experiment, submission, manual.
    scores = []
    file_paths = []
    sub_ids = []
    notes = []
    family_labels = []
    timestamps = []
    for _, r in table.iterrows():
        scores.append(_coalesce(r.get("public_lb_score_manual", pd.NA), r.get("public_lb_score_submission", pd.NA), r.get("public_lb_score", pd.NA)))
        file_paths.append(_coalesce(r.get("file_path_manual", pd.NA), r.get("file_path", pd.NA)))
        sub_ids.append(_coalesce(r.get("submission_id_manual", pd.NA), r.get("submission_id", pd.NA)))
        notes.append(" | ".join(str(x) for x in [r.get("notes", ""), r.get("submission_notes", ""), r.get("notes_manual", "")] if pd.notna(x) and str(x)))
        family_labels.append(_coalesce(r.get("chosen_family_label_manual", pd.NA), r.get("chosen_family_label", pd.NA), r.get("model_family", "")))
        timestamps.append(_coalesce(r.get("submission_timestamp_manual", pd.NA), r.get("submission_timestamp", pd.NA)))
    table["public_lb_score"] = pd.to_numeric(pd.Series(scores), errors="coerce")
    table["file_path"] = file_paths
    table["submission_id"] = sub_ids
    table["notes"] = notes
    table["chosen_family_label"] = family_labels
    table["submission_timestamp"] = timestamps
    table["public_minus_oof"] = table["public_lb_score"] - pd.to_numeric(table["OOF macro-F1"], errors="coerce")
    return table
