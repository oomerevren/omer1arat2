"""Risk flags and final strategic labels for public/OOF intelligence."""
from __future__ import annotations

from typing import Any
import pandas as pd

RISK_FLAGS = [
    "PUBLIC_UP_OOF_DOWN", "PUBLIC_UP_CLASS0_DOWN", "THRESHOLD_FRAGILE", "SPLITTER_INCONSISTENT",
    "SEGMENT_COLLAPSE_RISK", "ENSEMBLE_OVERFIT_RISK", "DENSE_ARTIFACT_RISK", "RETRIEVAL_FEATURE_DRIFT",
    "PRIVATE_UNSAFE_CANDIDATE",
]


def _truthy(v: Any) -> bool:
    return str(v).lower() in {"true", "1", "yes", "y"}


def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["OOF macro-F1"] = pd.to_numeric(out.get("OOF macro-F1"), errors="coerce")
    out["class0 F1"] = pd.to_numeric(out.get("class0 F1"), errors="coerce")
    out["public_lb_score"] = pd.to_numeric(out.get("public_lb_score"), errors="coerce")
    out["public_minus_oof"] = out["public_lb_score"] - out["OOF macro-F1"]
    # Baseline-relative deltas are computed in chronological/table order if possible.
    out["oof_delta_vs_prev"] = out["OOF macro-F1"].diff()
    out["class0_delta_vs_prev"] = out["class0 F1"].diff()
    out["public_delta_vs_prev"] = out["public_lb_score"].diff()
    return out


def flags_for_row(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    public_delta = row.get("public_delta_vs_prev")
    oof_delta = row.get("oof_delta_vs_prev")
    c0_delta = row.get("class0_delta_vs_prev")
    if pd.notna(public_delta) and pd.notna(oof_delta) and float(public_delta) > 0 and float(oof_delta) < 0:
        flags.append("PUBLIC_UP_OOF_DOWN")
    if pd.notna(public_delta) and pd.notna(c0_delta) and float(public_delta) > 0 and float(c0_delta) < 0:
        flags.append("PUBLIC_UP_CLASS0_DOWN")
    if _truthy(row.get("threshold_fragility", False)):
        flags.append("THRESHOLD_FRAGILE")
    seed_std = row.get("seed_std")
    if pd.notna(seed_std) and str(seed_std) != "" and float(seed_std) > 0.015:
        flags.append("SPLITTER_INCONSISTENT")
    fam = str(row.get("model_family", "")).lower()
    comp = " ".join(str(row.get(k, "")) for k in ["changed_component", "experiment_name", "notes"]).lower()
    if "ensemble" in fam and pd.notna(row.get("public_minus_oof")) and float(row.get("public_minus_oof")) > 0.02:
        flags.append("ENSEMBLE_OVERFIT_RISK")
    if "dense" in fam or "dense" in comp:
        flags.append("DENSE_ARTIFACT_RISK")
    if "retrieval" in fam or "retrieval" in comp:
        flags.append("RETRIEVAL_FEATURE_DRIFT")
    if pd.notna(row.get("public_minus_oof")) and float(row.get("public_minus_oof")) > 0.03:
        flags.append("PRIVATE_UNSAFE_CANDIDATE")
    return list(dict.fromkeys(flags))


def strategic_status(row: dict[str, Any]) -> str:
    flags = set(str(row.get("risk_flag", "")).split("|")) if row.get("risk_flag") else set()
    public = row.get("public_lb_score")
    oof = row.get("OOF macro-F1")
    c0 = row.get("class0 F1")
    if "PRIVATE_UNSAFE_CANDIDATE" in flags or "PUBLIC_UP_OOF_DOWN" in flags or "PUBLIC_UP_CLASS0_DOWN" in flags:
        return "public_optimistic"
    if len(flags & {"DENSE_ARTIFACT_RISK", "RETRIEVAL_FEATURE_DRIFT", "ENSEMBLE_OVERFIT_RISK", "THRESHOLD_FRAGILE", "SPLITTER_INCONSISTENT"}) >= 2:
        return "high_risk_experimental"
    if pd.notna(oof) and pd.notna(c0) and not flags:
        return "private_safe"
    if pd.notna(oof) and len(flags) <= 1:
        return "balanced_candidate"
    if pd.isna(oof) and pd.notna(public):
        return "do_not_use_for_final"
    return "high_risk_experimental"


def apply_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = add_delta_columns(df)
    out["risk_flag"] = ["|".join(flags_for_row(r)) for r in out.to_dict("records")]
    out["strategic_status"] = [strategic_status(r) for r in out.to_dict("records")]
    return out
