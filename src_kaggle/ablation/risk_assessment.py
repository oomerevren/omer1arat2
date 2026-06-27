"""Risk labels for ablation results."""
from __future__ import annotations

from typing import Any


def assess_risk(row: dict[str, Any], baseline_macro: float | None = None, baseline_class0: float | None = None) -> str:
    status = str(row.get("status", "completed"))
    if status not in {"completed", "ok"}:
        return "needs_fold_safe_recheck"
    note = str(row.get("note", "")).lower()
    cat = str(row.get("category", ""))
    changed = str(row.get("changed_component", "")).lower()
    fragile = str(row.get("threshold_fragility", "")).lower() in {"true", "1", "yes"}
    seed_std = float(row.get("seed_std", 0) or 0)
    macro = float(row.get("OOF macro-F1", row.get("oof_macro_f1", 0)) or 0)
    class0 = float(row.get("class0 F1", row.get("class0_f1", 0)) or 0)
    delta = 0.0 if baseline_macro is None else macro - baseline_macro
    c0_delta = 0.0 if baseline_class0 is None else class0 - baseline_class0

    if "public" in note and delta < 0:
        return "high_risk_public_only"
    if cat in {"retrieval", "dense", "negative"} and ("dense" in changed or "retrieval" in changed) and "fold" not in note:
        if delta > 0.002:
            return "needs_fold_safe_recheck"
    if fragile or seed_std > 0.015:
        return "medium_risk_monitor"
    if delta > 0.003 and c0_delta >= -0.002:
        return "low_risk_keep"
    if delta < -0.003 or c0_delta < -0.006:
        return "artifact_suspected" if cat in {"retrieval", "dense", "negative"} else "medium_risk_monitor"
    return "medium_risk_monitor"


def keep_risky_drop(row: dict[str, Any], baseline_macro: float | None = None) -> str:
    risk = str(row.get("risk_flag", ""))
    status = str(row.get("status", "completed"))
    if status != "completed":
        return "Good but risky"
    macro = float(row.get("OOF macro-F1", 0) or 0)
    base = baseline_macro if baseline_macro is not None else macro
    delta = macro - base
    if risk == "low_risk_keep" and delta >= 0:
        return "Keep no matter what"
    if risk in {"needs_fold_safe_recheck", "medium_risk_monitor", "high_risk_public_only"} and delta >= 0:
        return "Good but risky"
    if abs(delta) < 0.002:
        return "Segment benefit only"
    return "Drop from final pipeline"
