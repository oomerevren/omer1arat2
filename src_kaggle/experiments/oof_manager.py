"""Standard OOF artifact management for experiments."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import pandas as pd

REQUIRED_OOF_COLUMNS = ["id", "term_id", "item_id", "label", "fold", "proba", "model_name", "experiment_name"]


def standardize_oof(oof: pd.DataFrame, experiment_name: str, model_name: str, best_threshold: float) -> pd.DataFrame:
    out = oof.copy()
    out["experiment_name"] = experiment_name
    out["model_name"] = out.get("model_name", model_name)
    out["pred_default"] = (out["proba"] >= 0.5).astype(int)
    out["pred_best_threshold"] = (out["proba"] >= best_threshold).astype(int)
    missing = [c for c in REQUIRED_OOF_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"OOF missing required columns: {missing}")
    return out


def save_experiment_oof(oof: pd.DataFrame, experiment_dir: str | Path) -> str:
    path = Path(experiment_dir) / "oof_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(path, index=False)
    return str(path)


def snapshot_config(config: dict, experiment_dir: str | Path) -> str:
    path = Path(experiment_dir) / "config_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path)


def copy_if_exists(src: str | Path, dst: str | Path) -> str | None:
    src = Path(src); dst = Path(dst)
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)
