"""Risk and segment profiling for final submission families."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def family_risk_label(family_name: str, row: dict) -> str:
    name = family_name.lower()
    if "balanced" in name:
        return "safest_balanced"
    if "defensive" in name or "class0" in name:
        return "private_defensive"
    if "aggressive" in name or "semantic" in name:
        return "semantic_aggressive"
    flags = str(row.get("risk_flag", ""))
    if "PUBLIC" in flags or "PRIVATE_UNSAFE" in flags:
        return "public_optimistic"
    return "high_risk_experimental"


def build_family_segment_scores(families: pd.DataFrame, out_path: str | Path = "reports/final/final_family_segment_scores.csv") -> pd.DataFrame:
    # Real segment scores require family OOF predictions.  Until those exist, write
    # an explicit behavior map so the final protocol is auditable.
    rows = []
    defaults = {
        "family_A_balanced": ("balanced across short/brand/attribute segments", "no known segment edge without OOF"),
        "family_B_defensive": ("attribute/gender/age conflict and class0-heavy segments", "may under-recall semantic positives"),
        "family_C_aggressive": ("semantic/long-tail/dense-hard recall segments", "higher false-positive/private risk"),
    }
    for _, f in families.iterrows():
        strong, weak = defaults.get(f["family_name"], ("unknown", "unknown"))
        for seg in ["short_query","brand_heavy","attribute_heavy","category_heavy","same_category_hard","same_brand","dense_hard","gender_cue","age_cue","unknown_metadata"]:
            rows.append({"family_name": f["family_name"], "segment": seg, "macro_f1": pd.NA, "class0_f1": pd.NA, "class1_f1": pd.NA, "strength_note": strong, "weakness_note": weak, "status": "requires_family_oof"})
    table = pd.DataFrame(rows)
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True); table.to_csv(p, index=False)
    return table


def build_family_risk_flags(families: pd.DataFrame, out_path: str | Path = "reports/final/final_family_risk_flags.csv") -> pd.DataFrame:
    rows = []
    for _, f in families.iterrows():
        flags = []
        label = f.get("risk_label", f.get("public_private_risk_label", ""))
        if label == "semantic_aggressive": flags += ["DENSE_ARTIFACT_RISK", "PUBLIC_OPTIMISM_MONITOR"]
        if label == "private_defensive": flags += ["RECALL_LOSS_MONITOR", "THRESHOLD_HIGH_MONITOR"]
        if label == "safest_balanced": flags += ["DEFAULT_SAFE", "PUBLIC_GAIN_NOT_REQUIRED"]
        if str(f.get("status", "")).startswith("not_ready"):
            flags += ["MISSING_SUBMISSION_ARTEFACT", "REQUIRES_REAL_TEST_PREDICTIONS"]
        rows.append({"family_name": f["family_name"], "risk_label": label, "risk_flags": "|".join(flags), "notes": f.get("notes", "")})
    table = pd.DataFrame(rows)
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True); table.to_csv(p, index=False)
    return table
