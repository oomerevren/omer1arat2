"""
Ana eğitim orkestratörü — veri hazırlama, CV, distillation, pseudo-label, quantization.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.data.augmenter import DataAugmenter
from src.data.dataset import get_num_labels, load_train_test
from src.data.labels import normalize_labels
from src.data.negative_miner import NegativeMiner
from src.data.preprocessor import TextPreprocessor
from src.data.sample_generator import ensure_sample_data
from src.evaluation.threshold_search import search_multiclass_thresholds
from src.evaluation.validator import CrossValidator
from src.experiment.tracker import get_tracker
from src.inference.quantizer import apply_quantization
from src.models.cross_encoder import CrossEncoderModel
from src.models.ensemble import EnsembleModel
from src.training.distillation import run_distillation
from src.training.pseudo_labeler import PseudoLabeler
from src.utils.io import write_json


def preprocess_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    prep = TextPreprocessor.from_config(config)
    out = df.copy()
    q_col = config.get("data", {}).get("query_column", "search_query")
    t_col = config.get("data", {}).get("product_text_column", "product_name")
    if q_col in out.columns:
        out[q_col] = out[q_col].astype(str).map(prep.preprocess)
    if t_col in out.columns:
        out[t_col] = out[t_col].astype(str).map(prep.preprocess)
    return out


def apply_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    aug_cfg = config.get("data", {}).get("augmentation", {})
    if not aug_cfg.get("enabled", False):
        return df
    q_col = config.get("data", {}).get("query_column", "search_query")
    augmenter = DataAugmenter()
    return augmenter.augment_dataframe(
        df, q_col, n_variants=aug_cfg.get("max_per_query", 2)
    )


def apply_negative_mining(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    nm_cfg = config.get("data", {}).get("negative_mining", {})
    if not nm_cfg.get("enabled", False):
        return df
    strategies = nm_cfg.get("strategies", {})
    miner = NegativeMiner(
        negatives_per_positive=nm_cfg.get("negatives_per_positive", 5),
        strategies=strategies,
        dense_model_name=nm_cfg.get("dense_model_name", "intfloat/multilingual-e5-large"),
    )
    return miner.mine(df)


def apply_pseudo_labeling(
    train_df: pd.DataFrame,
    config: Dict[str, Any],
    teacher: CrossEncoderModel,
) -> pd.DataFrame:
    pl_cfg = config.get("pseudo_labeling", {})
    if not pl_cfg.get("enabled", False):
        return train_df

    unlabeled_path = pl_cfg.get("unlabeled_path")
    if unlabeled_path and os.path.exists(unlabeled_path):
        from src.data.dataset import load_csv

        unlabeled = load_csv(unlabeled_path, config)
    else:
        unlabeled = train_df.copy()

    labeler = PseudoLabeler(teacher)
    pseudo = labeler.generate_labels(unlabeled, config)
    pos_th = pl_cfg.get("positive_threshold", 0.99)
    neg_th = pl_cfg.get("negative_threshold", 0.01)

    if "pseudo_confidence" in pseudo.columns:
        high_conf = pseudo[
            (pseudo["pseudo_confidence"] >= pos_th)
            | (pseudo["pseudo_confidence"] <= neg_th)
        ]
        if len(high_conf) > 0:
            return pd.concat([train_df, high_conf], ignore_index=True)
    return train_df


def build_model(config: Dict[str, Any]):
    model_type = config.get("model", {}).get("type", "cross_encoder")
    if model_type == "ensemble":
        return EnsembleModel(config)
    ce_cfg = config.get("model", {}).get("cross_encoder", {})
    return CrossEncoderModel(
        model_name=ce_cfg.get("model_name", "dbmdz/distilbert-base-turkish-cased"),
        num_labels=get_num_labels(config),
        max_length=ce_cfg.get("max_length", 256),
    )


def _prepare_data(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    train_df = normalize_labels(train_df, config)
    if val_df is not None:
        val_df = normalize_labels(val_df, config)

    train_df = preprocess_dataframe(train_df, config)
    if val_df is not None:
        val_df = preprocess_dataframe(val_df, config)

    train_df = apply_augmentation(train_df, config)

    mode = config.get("experiment", {}).get("mode", "final")
    if mode == "kaggle" or config.get("data", {}).get("negative_mining", {}).get("enabled"):
        train_df = apply_negative_mining(train_df, config)
        train_df = normalize_labels(train_df, config)

    return train_df, val_df


def _run_cross_validation(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """CV metriklerini hesapla (eğitim öncesi stabilite raporu)."""

    def train_fn(tr_fold, val_fold, cfg):
        m = build_model(cfg)
        if isinstance(m, EnsembleModel):
            m.fit(tr_fold, cfg)
        else:
            m.train(tr_fold, cfg, val_fold)
        return m

    def eval_fn(model, val_fold, cfg):
        return model.evaluate(val_fold, cfg)

    cv = CrossValidator(config)
    report = cv.validate(df, train_fn, eval_fn)
    return {
        "cv_mean_f1": report.overall_mean_f1,
        "cv_std_f1": report.overall_std_f1,
        "cv_stability": report.stability_score,
    }


def _evaluate_and_threshold(
    model,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
    metrics: Dict[str, float],
) -> Dict[str, float]:
    val_metrics = model.evaluate(val_df, config)
    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

    label_col = config.get("data", {}).get("label_column", "is_relevant")
    cat_col = config.get("data", {}).get("category_column", "category")
    probs = model.predict_proba(val_df, config)
    y_true = val_df[label_col].astype(int).values

    thresholds, th_f1, cat_thresholds = search_multiclass_thresholds(
        y_true, probs, config, val_df=val_df, category_col=cat_col
    )
    metrics["threshold_f1"] = th_f1
    for k, v in thresholds.items():
        metrics[f"threshold_{k}"] = v
    if cat_thresholds:
        metrics["category_threshold_count"] = float(len(cat_thresholds))

    return metrics


def _save_metrics_json(metrics: Dict[str, float], save_path: Path) -> None:
    out = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    write_json(save_path / "metrics.json", out)


def train_model(
    config: Dict[str, Any],
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
) -> Tuple[Any, Dict[str, float]]:
    if train_df is None:
        ensure_sample_data(config)
        train_df, test_df = load_train_test(config)
        if val_df is None and test_df is not None:
            val_df = test_df

    train_df, val_df = _prepare_data(train_df, val_df, config)
    metrics: Dict[str, float] = {}

    # Cross-validation (final mod, config ile aktif)
    val_cfg = config.get("validation", {})
    sk_cfg = val_cfg.get("methods", {}).get("stratified_kfold", {})
    if sk_cfg.get("enabled") and config.get("training", {}).get("run_cv_before_train", True):
        try:
            metrics.update(_run_cross_validation(train_df, config))
        except Exception as exc:
            metrics["cv_error"] = 1.0
            print(f"[CV] Uyarı: cross-validation başarısız, eğitim devam ediyor: {exc}")
            traceback.print_exc()

    # Distillation veya standart eğitim
    if config.get("distillation", {}).get("enabled", False):
        model, d_metrics = run_distillation(config, train_df, val_df)
        metrics.update(d_metrics)
    else:
        model = build_model(config)
        if isinstance(model, EnsembleModel):
            metrics.update(model.fit(train_df, config))
        else:
            metrics.update(model.train(train_df, config, val_df))

    # Pseudo-labeling (teacher = mevcut model)
    if config.get("pseudo_labeling", {}).get("enabled", False):
        if isinstance(model, CrossEncoderModel):
            augmented = apply_pseudo_labeling(train_df, config, model)
            if len(augmented) > len(train_df):
                metrics.update(model.train(augmented, config, val_df))

    if val_df is not None and len(val_df) > 0:
        metrics = _evaluate_and_threshold(model, val_df, config, metrics)

    return model, metrics


def save_model(
    model,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    output_dir: Optional[str] = None,
) -> str:
    output_dir = output_dir or config.get("experiment", {}).get("output_dir", "./experiments/outputs")
    exp_name = config.get("experiment", {}).get("name", "baseline")
    save_path = Path(output_dir) / exp_name
    save_path.mkdir(parents=True, exist_ok=True)

    if isinstance(model, CrossEncoderModel):
        model.save(save_path / "cross_encoder")
        if config.get("quantization", {}).get("enabled", False):
            # model wrapper'ı geçiyoruz: tokenizer + max_length erişimi için
            q_paths = apply_quantization(
                config, model, str(save_path / "quantized")
            )
            if q_paths and metrics is not None:
                metrics["quantization_paths"] = float(len(q_paths))
    elif isinstance(model, EnsembleModel) and model.cross_encoder:
        model.cross_encoder.save(save_path / "cross_encoder")

    if metrics:
        _save_metrics_json(metrics, save_path)

    return str(save_path)


def resolve_model_path(config: Dict[str, Any]) -> Path:
    """Submission/API için tutarlı model yolu."""
    output_dir = config.get("experiment", {}).get("output_dir", "./experiments/outputs")
    exp_name = config.get("experiment", {}).get("name", "baseline")
    return Path(output_dir) / exp_name / "cross_encoder"


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    tracker = get_tracker(
        experiment_name=config.get("tracking", {}).get("experiment_name", "teknofest2026"),
        tracking_uri=config.get("tracking", {}).get("mlflow_uri"),
    )
    exp_name = config.get("experiment", {}).get("name", "baseline")

    with tracker.start_run(exp_name) as run:
        tracker.log_config(config)
        model, metrics = train_model(config)
        tracker.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        save_path = save_model(model, config, metrics)
        tracker.log_model_info(
            model_type=config.get("model", {}).get("type", "cross_encoder"),
            model_name=config.get("model", {}).get("cross_encoder", {}).get("model_name", ""),
        )
        return {"run_id": run.run_id, "metrics": metrics, "save_path": save_path}
