"""Evidence collection for championship readiness audit."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd


def exists(path: str | Path) -> bool:
    return Path(path).exists()


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def collect_evidence() -> dict[str, Any]:
    files = {
        "war_mode_config": "configs/kaggle/war_mode.yaml",
        "final_base_config": "configs/kaggle/final/shared_final_base.yaml",
        "family_A_config": "configs/kaggle/final/family_A_balanced.yaml",
        "family_B_config": "configs/kaggle/final/family_B_defensive.yaml",
        "family_C_config": "configs/kaggle/final/family_C_aggressive.yaml",
        "manifest": "reports/final/final_artifact_manifest.json",
        "release_validation": "reports/final/final_release_validation_report.json",
        "submission_registry": "reports/submissions/submission_registry.csv",
        "ablation_master": "reports/ablation/ablation_master_table.csv",
        "leaderboard_table": "reports/leaderboard/oof_public_correlation.csv",
        "final_families": "reports/final/final_submission_families.csv",
        "final_candidate_pool": "reports/final/final_candidate_pool.csv",
        "readme": "README.md",
        "competition_freeze": "docs/competition_freeze.md",
        "submission_checklist": "docs/submission_day_checklist.md",
    }
    tests = list(Path("tests_kaggle").glob("test_*.py")) if Path("tests_kaggle").exists() else []
    release = read_json(files["release_validation"])
    manifest = read_json(files["manifest"])
    ablation = read_csv(files["ablation_master"])
    leaderboard = read_csv(files["leaderboard_table"])
    families = read_csv(files["final_families"])
    submission_paths = [
        "artifacts/final/submissions/family_A_balanced_submission.csv",
        "artifacts/final/submissions/family_B_defensive_submission.csv",
        "artifacts/final/submissions/family_C_aggressive_submission.csv",
    ]
    return {
        "files": {k: {"path": v, "exists": exists(v)} for k, v in files.items()},
        "release_validation": release,
        "manifest": manifest,
        "ablation_rows": int(len(ablation)),
        "ablation_completed_rows": int(ablation.get("status", pd.Series(dtype=str)).eq("completed").sum()) if not ablation.empty else 0,
        "leaderboard_rows": int(len(leaderboard)),
        "leaderboard_public_points": int(pd.to_numeric(leaderboard.get("public_lb_score", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not leaderboard.empty else 0,
        "family_rows": int(len(families)),
        "family_statuses": families[["family_name", "status", "public_private_risk_label"]].to_dict("records") if not families.empty and {"family_name", "status", "public_private_risk_label"}.issubset(families.columns) else [],
        "tests_kaggle_count": len(tests),
        "submission_paths": {p: exists(p) for p in submission_paths},
        "key_dirs": {p: exists(p) for p in ["src_kaggle", "scripts_kaggle", "configs/kaggle/final", "artifacts/final", "reports/final", "docs"]},
    }
