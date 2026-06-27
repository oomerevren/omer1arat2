"""Checkpoint/provenance helpers for transformer cross-encoder."""
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone


def fold_dir(root: str | Path, experiment_name: str, fold: int) -> Path:
    return Path(root) / experiment_name / f"fold_{fold}"


def write_training_config(path: str | Path, config: dict) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {**config, "written_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path)


def write_metrics(path: str | Path, metrics: dict) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path)
