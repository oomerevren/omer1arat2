from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.models.cross_encoder_model import CrossEncoderModel
from src_kaggle.models.oof import OOFResult, build_oof_frame, find_best_threshold, make_folds, write_json
from src_kaggle.models.transformer_trainer import TransformerCrossEncoder
from src_kaggle.validation.threshold_tuning import metrics_at_threshold


def train_cross_encoder_oof(df: pd.DataFrame, cfg: dict, artifact_dir: str | None = None, experiment_name: str = "cross_encoder") -> OOFResult:
    seed = int(cfg.get("seed", 42)); n_folds = int(cfg.get("n_folds", 5)); backend = cfg.get("backend", "sklearn_text")
    artifact_dir = artifact_dir or ("artifacts/models/cross_encoder_transformer" if backend == "transformers" else "artifacts/models/cross_encoder")
    if "fold" not in df.columns:
        df = df.copy(); df["fold"] = make_folds(df, n_folds, seed)
    oof = np.zeros(len(df)); fold_scores=[]; fold_reports=[]; Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    for fold in sorted(df["fold"].unique()):
        tr = df[df["fold"] != fold].reset_index(drop=True); va = df[df["fold"] == fold].reset_index(drop=True)
        start = time.time()
        if backend == "transformers":
            fold_dir = Path(artifact_dir) / experiment_name / f"fold_{fold}"
            model = TransformerCrossEncoder({**cfg, "seed": seed + int(fold)})
            train_metrics = model.fit(tr, va, fold_dir)
            pred = model.predict_proba(va)
        else:
            model = CrossEncoderModel(
                backend=backend, model_name=cfg.get("model_name", "dbmdz/distilbert-base-turkish-cased"),
                text_format_version=cfg.get("text_format_version", "full_v1"), seed=seed+int(fold), params=cfg.get("params", {})
            ).fit(tr, tr[SCHEMA.label].astype(int))
            pred = model.predict_proba(va)
            fold_dir = Path(artifact_dir) / f"{backend}_fold{fold}"
            model.save(str(fold_dir) + ".joblib")
            train_metrics = {}
        oof[df[df["fold"] == fold].index] = pred
        th, score = find_best_threshold(va[SCHEMA.label].astype(int), pred); fold_scores.append(score)
        m = metrics_at_threshold(va[SCHEMA.label].astype(int), pred, th)
        m.update({"fold": int(fold), "fold_best_threshold": th, "training_duration_sec": float(time.time()-start), "artifact_dir": str(fold_dir), "train_metrics": train_metrics})
        fold_reports.append(m)
    best_th, macro = find_best_threshold(df[SCHEMA.label].astype(int), oof)
    oof_df = build_oof_frame(df, oof, "cross_encoder_transformer" if backend == "transformers" else "cross_encoder", "fold")
    oof_path = str(Path("artifacts/oof") / ("cross_encoder_transformer_oof.csv" if backend == "transformers" else "cross_encoder_oof.csv")); Path(oof_path).parent.mkdir(parents=True, exist_ok=True); oof_df.to_csv(oof_path, index=False)
    report = {"model":"cross_encoder", "backend": backend, "model_name": cfg.get("model_name"), "tokenizer_name": cfg.get("tokenizer_name"), "text_format_version": cfg.get("text_format_version", "full_v1"), "max_length": cfg.get("max_length", 256), "macro_f1": macro, "best_threshold": best_th, "fold_scores": fold_scores, "fold_reports": fold_reports, "artifact_dir": artifact_dir, "oof_path": oof_path}
    report_path = "reports/models/cross_encoder_transformer_cv_report.json" if backend == "transformers" else "reports/models/cross_encoder_cv_report.json"; write_json(report, report_path)
    if backend == "transformers":
        row = {"backbone": cfg.get("model_name"), "text_format": cfg.get("text_format_version", "full_v1"), "max_length": cfg.get("max_length", 256), "OOF macro-F1": macro, "class0 F1": metrics_at_threshold(df[SCHEMA.label], oof, best_th)["class0_f1"], "class1 F1": metrics_at_threshold(df[SCHEMA.label], oof, best_th)["class1_f1"], "threshold": best_th, "train_time": sum(fr.get("training_duration_sec",0) for fr in fold_reports), "note": "transformers backend"}
        path=Path("reports/models/cross_encoder_transformer_ablation.csv"); path.parent.mkdir(parents=True, exist_ok=True)
        old=pd.read_csv(path) if path.exists() else pd.DataFrame(); pd.concat([old,pd.DataFrame([row])],ignore_index=True).to_csv(path,index=False)
        token_stats=[fr.get("train_metrics",{}).get("token_stats") for fr in fold_reports if fr.get("train_metrics",{}).get("token_stats")]
        Path("reports/models/cross_encoder_transformer_token_stats.json").write_text(json.dumps(token_stats,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    return OOFResult("cross_encoder_transformer" if backend == "transformers" else "cross_encoder", oof_path, artifact_dir, best_th, macro, fold_scores, report_path)
