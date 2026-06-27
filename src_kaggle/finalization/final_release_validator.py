"""Validate final release/freeze readiness."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any

import yaml

FAMILY_NAMES = ["family_A_balanced", "family_B_defensive", "family_C_aggressive"]
REQUIRED_CONFIG_FIELDS = ["final_mode", "frozen", "family_name", "artifact_paths", "submission_output_path", "blend_weights", "threshold", "source_experiment_ids", "used_data_version", "negative_mining_version", "retrieval_version", "feature_version", "validation_version"]
REQUIRED_METADATA_FIELDS = ["family_name", "risk_label", "models_used", "blend_weights", "threshold", "created_at", "status"]


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def validate_final_release(
    config_dir: str | Path = "configs/kaggle/final",
    final_root: str | Path = "artifacts/final",
    reports_dir: str | Path = "reports/final",
) -> dict[str, Any]:
    config_dir = Path(config_dir); final_root = Path(final_root); reports_dir = Path(reports_dir)
    errors=[]; warnings=[]; family_checks={}
    manifest_path = reports_dir / "final_artifact_manifest.json"
    manifest = _json(manifest_path)
    if not manifest_path.exists(): errors.append("missing final_artifact_manifest.json")
    base_path = config_dir / "shared_final_base.yaml"
    if not base_path.exists(): errors.append("missing shared_final_base.yaml")
    for fam in FAMILY_NAMES:
        cfg_path = config_dir / f"{fam}.yaml"
        meta_path = final_root / "families" / fam / "metadata.json"
        sub_path = final_root / "submissions" / f"{fam}_submission.csv"
        cfg = _yaml(cfg_path); meta = _json(meta_path)
        miss_cfg=[k for k in REQUIRED_CONFIG_FIELDS if k not in cfg]
        miss_meta=[k for k in REQUIRED_METADATA_FIELDS if k not in meta]
        fam_errors=[]; fam_warnings=[]
        if not cfg_path.exists(): fam_errors.append("missing family config")
        if not meta_path.exists(): fam_errors.append("missing standardized metadata")
        if miss_cfg: fam_errors.append(f"missing config fields: {miss_cfg}")
        if miss_meta: fam_warnings.append(f"missing metadata fields: {miss_meta}")
        if cfg.get("final_mode") is not True: fam_errors.append("final_mode is not true")
        if cfg.get("frozen") is not True: fam_errors.append("frozen is not true")
        if not sub_path.exists(): fam_warnings.append("submission.csv not materialized yet; requires official test predictions")
        validation_report = meta.get("validation_report")
        if sub_path.exists() and (not validation_report or not Path(validation_report).exists()): fam_errors.append("submission exists but validation report missing")
        family_checks[fam] = {"config_path": str(cfg_path), "metadata_path": str(meta_path), "submission_path": str(sub_path), "errors": fam_errors, "warnings": fam_warnings, "release_ready": not fam_errors and sub_path.exists()}
        errors.extend([f"{fam}: {e}" for e in fam_errors]); warnings.extend([f"{fam}: {w}" for w in fam_warnings])
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "release_ready": len(errors)==0 and all(v["release_ready"] for v in family_checks.values()), "metadata_lock_ready": len(errors)==0, "errors": errors, "warnings": warnings, "family_checks": family_checks, "manifest_status": manifest.get("final_release_status", "unknown")}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "final_release_validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines=["# Final Release Validation", "", f"- release_ready: `{report['release_ready']}`", f"- metadata_lock_ready: `{report['metadata_lock_ready']}`", f"- errors: {len(errors)}", f"- warnings: {len(warnings)}", "", "## Errors", ""]
    lines += [f"- {e}" for e in errors] or ["_none_"]
    lines += ["", "## Warnings", ""]
    lines += [f"- {w}" for w in warnings] or ["_none_"]
    (reports_dir / "final_release_validation.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report
