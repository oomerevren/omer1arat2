"""Build final artifact manifest for race-day provenance."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any

import pandas as pd
import yaml

from src_kaggle.finalization.config_freezer import sha256_file

FAMILY_NAMES = ["family_A_balanced", "family_B_defensive", "family_C_aggressive"]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_final_artifact_manifest(
    config_dir: str | Path = "configs/kaggle/final",
    final_root: str | Path = "artifacts/final",
    reports_dir: str | Path = "reports/final",
    submission_registry: str | Path = "reports/submissions/submission_registry.csv",
) -> dict[str, Any]:
    config_dir = Path(config_dir); final_root = Path(final_root); reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    families = {}
    for fam in FAMILY_NAMES:
        cfg_path = config_dir / f"{fam}.yaml"
        std_dir = final_root / "families" / fam
        metadata_path = std_dir / "metadata.json"
        cfg = _yaml(cfg_path)
        meta = _json(metadata_path)
        sub_path = final_root / "submissions" / f"{fam}_submission.csv"
        families[fam] = {
            "config_path": str(cfg_path),
            "config_exists": cfg_path.exists(),
            "config_sha256": sha256_file(cfg_path) if cfg_path.exists() else None,
            "final_mode": cfg.get("final_mode"),
            "frozen": cfg.get("frozen"),
            "artifact_dir": str(std_dir),
            "metadata_path": str(metadata_path),
            "metadata_exists": metadata_path.exists(),
            "submission_path": str(sub_path),
            "submission_exists": sub_path.exists(),
            "models_used": meta.get("models_used", []),
            "source_experiment_ids": meta.get("experiment_ids", cfg.get("source_experiment_ids", [])),
            "source_model_artifact_paths": meta.get("source_model_artifact_paths", []),
            "source_oof_reports": meta.get("source_oof_reports", []),
            "blend_weights": meta.get("blend_weights", cfg.get("blend_weights", {})),
            "threshold": meta.get("threshold", cfg.get("threshold")),
            "risk_label": meta.get("risk_label", cfg.get("risk_label")),
            "OOF macro-F1": meta.get("OOF macro-F1"),
            "class0 F1": meta.get("class0 F1"),
            "class1 F1": meta.get("class1 F1"),
            "status": meta.get("status", "metadata_missing"),
            "validation_report": meta.get("validation_report", ""),
        }
    subreg = Path(submission_registry)
    registry_rows = 0
    registry_tail = []
    if subreg.exists() and subreg.stat().st_size > 0:
        try:
            df = pd.read_csv(subreg)
            registry_rows = len(df)
            registry_tail = df.tail(10).to_dict("records")
        except Exception:
            pass
    manifest = {
        "manifest_version": "final_artifact_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competition": "TEKNOFEST 2026 E-Ticaret Kaggle",
        "final_release_status": "locked_metadata_pending_real_submissions" if not all(f["submission_exists"] for f in families.values()) else "submission_ready",
        "families": families,
        "registries": {"submission_registry": str(subreg), "submission_registry_exists": subreg.exists(), "submission_registry_rows": registry_rows, "submission_registry_tail": registry_tail, "experiment_registry": "reports/experiments/experiment_registry.csv", "ablation_master": "reports/ablation/ablation_master_table.csv", "leaderboard_table": "reports/leaderboard/oof_public_correlation.csv"},
        "active_final_paths": ["src_kaggle/", "scripts_kaggle/", "configs/kaggle/final/", "artifacts/final/", "reports/final/"],
        "do_not_use_as_final": ["src/", "scripts/", "src_hackathon/", "scripts_hackathon/", "configs/kaggle/experiments/"],
    }
    out = reports_dir / "final_artifact_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (final_root / "manifests").mkdir(parents=True, exist_ok=True)
    (final_root / "manifests" / "final_artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest
