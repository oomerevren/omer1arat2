"""GO/NO-GO decision logic."""
from __future__ import annotations
from typing import Any
import pandas as pd


def decide_go_no_go(component_table: pd.DataFrame, evidence: dict[str, Any]) -> dict[str, Any]:
    red = component_table[component_table["status"].eq("red")]
    high_blockers = red[(red["private_lb_impact"].isin(["yüksek", "çok yüksek", "high"])) | (red["submission_day_impact"].isin(["yüksek", "çok yüksek", "high"]))]
    release = evidence.get("release_validation", {})
    release_ready = bool(release.get("release_ready", False))
    metadata_lock_ready = bool(release.get("metadata_lock_ready", False))
    actual_submissions_exist = all(evidence.get("submission_paths", {}).values()) if evidence.get("submission_paths") else False
    if not actual_submissions_exist or not release_ready:
        decision = "NO_GO"
        rationale = "Final metadata/config freeze is ready, but actual validated final submission artefacts are not materialized. Immediate Kaggle submission is not safe."
    elif len(high_blockers) > 0:
        decision = "NO_GO"
        rationale = "At least one high-impact red blocker remains."
    elif len(red) > 0 or not metadata_lock_ready:
        decision = "GO_WITH_RISKS"
        rationale = "No submission artefact blocker, but some red/yellow risk remains."
    else:
        decision = "GO"
        rationale = "All critical release and component checks are green/yellow with validated submissions."
    return {
        "final_decision": decision,
        "rationale": rationale,
        "blocking_issues": red[["component_name", "open_gaps", "recommended_action"]].to_dict("records"),
        "acceptable_risks": component_table[component_table["status"].eq("yellow")][["component_name", "open_gaps", "recommended_action"]].to_dict("records"),
        "required_actions_before_submission": [
            "Materialize family submission.csv files from official test predictions.",
            "Run submission validator and ensure reports are present.",
            "Re-run package_final_families.py and validate_final_release.py until release_ready=true.",
            "Run at least one real OOF/ablation cycle on official data before trusting model-family decisions.",
        ] if decision != "GO" else [],
    }
