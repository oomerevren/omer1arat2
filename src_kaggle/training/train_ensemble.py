from __future__ import annotations

from pathlib import Path
import pandas as pd

from src_kaggle.models.ensemble import WeightedAverageEnsemble
from src_kaggle.models.oof import write_json


def train_weighted_ensemble(oof_paths: dict[str, str], cfg: dict, artifact_path: str = "artifacts/models/ensemble/weighted_average.joblib") -> dict:
    frames = {name: pd.read_csv(path).sort_values(["id"]).reset_index(drop=True) for name, path in oof_paths.items()}
    ens = WeightedAverageEnsemble(weights=cfg.get("blend_weights"))
    report = ens.fit_oof(frames)
    ens.save(artifact_path)
    report["artifact_path"] = artifact_path
    report_path = "reports/models/ensemble_cv_report.json"; write_json(report, report_path)
    return report
