"""Final Kaggle config freeze utilities.

Final configs are treated as operational release artefacts, not experiment knobs.
This module stamps final_mode/frozen metadata and records checksums in a freeze
index so race-day changes are visible.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import hashlib
import json

import pandas as pd
import yaml

FAMILY_NAMES = ["family_A_balanced", "family_B_defensive", "family_C_aggressive"]


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def ensure_shared_final_base(config_dir: str | Path = "configs/kaggle/final") -> Path:
    config_dir = Path(config_dir); config_dir.mkdir(parents=True, exist_ok=True)
    base_path = config_dir / "shared_final_base.yaml"
    if not base_path.exists():
        base = {
            "final_mode": True,
            "frozen": True,
            "schema_version": "final_freeze_v1",
            "competition": "TEKNOFEST 2026 E-Ticaret Kaggle",
            "problem": {"type": "binary_pair_classification", "metric": "macro_f1", "submission_columns": ["id", "prediction"]},
            "active_zones": {"final": ["src_kaggle/", "scripts_kaggle/", "configs/kaggle/final/", "artifacts/final/", "reports/final/"], "experimental": ["configs/kaggle/experiments/", "reports/ablation/", "reports/leaderboard/"], "legacy": ["src/", "scripts/", "src_hackathon/", "scripts_hackathon/"]},
            "decision_priority": ["OOF macro-F1", "class0 F1", "splitter reliability", "threshold fragility", "seed stability", "segment collapse risk", "model family drift", "public LB"],
            "submission_guard": {"validator_required": True, "registry_required": True, "public_lb_is_last_signal": True},
        }
        _write_yaml(base_path, base)
    return base_path


def freeze_family_configs(
    config_dir: str | Path = "configs/kaggle/final",
    artifact_root: str | Path = "artifacts/final_submissions",
    reports_dir: str | Path = "reports/final",
) -> pd.DataFrame:
    config_dir = Path(config_dir); artifact_root = Path(artifact_root); reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    base_path = ensure_shared_final_base(config_dir)
    frozen_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for fam in FAMILY_NAMES:
        cfg_path = config_dir / f"{fam}.yaml"
        data = _read_yaml(cfg_path)
        metadata_path = artifact_root / fam / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        data.update({
            "final_mode": True,
            "frozen": True,
            "schema_version": "final_family_freeze_v1",
            "family_name": fam,
            "shared_final_base": str(base_path),
            "frozen_at": data.get("frozen_at") or frozen_at,
            "change_policy": "Do not edit directly. Create configs/kaggle/experiments/<name>.yaml, rerun validation, then promote via freeze script.",
            "artifact_paths": {
                "legacy_family_artifact_dir": str(artifact_root / fam),
                "standard_family_artifact_dir": f"artifacts/final/families/{fam}",
                "standard_submission_path": f"artifacts/final/submissions/{fam}_submission.csv",
                "metadata_json": str(metadata_path),
            },
            "submission_output_path": f"artifacts/final/submissions/{fam}_submission.csv",
            "source_metadata": metadata,
        })
        # Ensure explicit fields requested by prompt exist even before real OOF.
        data.setdefault("models_used", metadata.get("models_used", []))
        data.setdefault("ensemble_method", "weighted_average")
        data.setdefault("blend_weights", metadata.get("blend_weights", {}))
        data.setdefault("threshold", metadata.get("threshold"))
        data.setdefault("splitter_reference", "term_group_default_private_safe")
        data.setdefault("source_experiment_ids", metadata.get("experiment_ids", []))
        data.setdefault("used_data_version", "official_kaggle_data_contract_v1")
        data.setdefault("negative_mining_version", "war_mode_negative_mining_config")
        data.setdefault("retrieval_version", "war_mode_retrieval_config")
        data.setdefault("feature_version", "war_mode_feature_engineering_config")
        data.setdefault("validation_version", "term_group_oof_private_lb_simulation")
        _write_yaml(cfg_path, data)
        rows.append({
            "family_name": fam,
            "config_path": str(cfg_path),
            "exists": cfg_path.exists(),
            "final_mode": bool(data.get("final_mode")),
            "frozen": bool(data.get("frozen")),
            "sha256": sha256_file(cfg_path),
            "metadata_path": str(metadata_path),
            "metadata_exists": metadata_path.exists(),
            "status": metadata.get("status", "metadata_missing"),
        })
    rows.append({"family_name": "shared_final_base", "config_path": str(base_path), "exists": True, "final_mode": True, "frozen": True, "sha256": sha256_file(base_path), "metadata_path": "", "metadata_exists": "", "status": "base_config"})
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "final_config_freeze_index.csv", index=False)
    (reports_dir / "final_config_freeze_index.json").write_text(json.dumps(df.to_dict("records"), indent=2, ensure_ascii=False), encoding="utf-8")
    return df
