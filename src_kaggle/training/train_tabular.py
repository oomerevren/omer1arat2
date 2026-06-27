from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from src_kaggle.models.oof import OOFResult, build_oof_frame, find_best_threshold, make_folds, write_json
from src_kaggle.models.tabular_model import TabularModel


def train_tabular_oof(df: pd.DataFrame, items: pd.DataFrame | None, cfg: dict, artifact_dir: str = "artifacts/models/tabular") -> OOFResult:
    seed = int(cfg.get("seed", 42)); n_folds = int(cfg.get("n_folds", 5)); model_type = cfg.get("model_type", "hist_gradient_boosting")
    fp = FeaturePipeline(FeaturePipelineConfig.from_dict(cfg.get("features", {})), items=items)
    if "fold" not in df.columns:
        df = df.copy(); df["fold"] = make_folds(df, n_folds, seed)
    oof = np.zeros(len(df)); fold_scores=[]; importances=[]
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    for fold in sorted(df["fold"].unique()):
        tr = df[df["fold"] != fold]; va = df[df["fold"] == fold]
        Xtr = fp.transform(tr).features; Xva = fp.transform(va).features
        model = TabularModel(model_type=model_type, params=cfg.get("params", {}), seed=seed+int(fold)).fit(Xtr, tr[SCHEMA.label].astype(int))
        pred = model.predict_proba(Xva); oof[va.index] = pred
        th, score = find_best_threshold(va[SCHEMA.label].astype(int), pred); fold_scores.append(score)
        model.save(Path(artifact_dir)/f"tabular_fold{fold}.joblib")
        imp = model.feature_importance(); imp["fold"] = fold; importances.append(imp)
    best_th, macro = find_best_threshold(df[SCHEMA.label].astype(int), oof)
    oof_df = build_oof_frame(df, oof, "tabular", "fold")
    oof_path = str(Path("artifacts/oof/tabular_oof.csv")); Path(oof_path).parent.mkdir(parents=True, exist_ok=True); oof_df.to_csv(oof_path, index=False)
    if importances:
        pd.concat(importances).groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).to_csv("reports/models/tabular_feature_importance.csv", index=False)
    report = {"model":"tabular", "model_type": model_type, "macro_f1": macro, "best_threshold": best_th, "fold_scores": fold_scores, "artifact_dir": artifact_dir, "oof_path": oof_path}
    report_path = "reports/models/tabular_cv_report.json"; write_json(report, report_path)
    return OOFResult("tabular", oof_path, artifact_dir, best_th, macro, fold_scores, report_path)
