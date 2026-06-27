"""CSV experiment registry/catalog."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

REGISTRY_COLUMNS = [
    "created_at", "experiment_id", "experiment_name", "model_type", "backbone", "booster_type",
    "data_version", "negative_mining_version", "retrieval_version", "feature_version", "validation_version",
    "seed", "fold_count", "oof_macro_f1", "class0_f1", "class1_f1", "best_threshold",
    "threshold_fragile", "oof_path", "report_dir", "submission_note", "public_lb_score",
]


def append_experiment(record: dict, registry_path: str | Path = "reports/experiments/experiment_registry.csv") -> pd.DataFrame:
    path = Path(registry_path); path.parent.mkdir(parents=True, exist_ok=True)
    row = {c: record.get(c, "") for c in REGISTRY_COLUMNS}
    row["created_at"] = row.get("created_at") or datetime.now(timezone.utc).isoformat()
    df_old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=REGISTRY_COLUMNS)
    df = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def load_registry(registry_path: str | Path = "reports/experiments/experiment_registry.csv") -> pd.DataFrame:
    path = Path(registry_path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=REGISTRY_COLUMNS)
