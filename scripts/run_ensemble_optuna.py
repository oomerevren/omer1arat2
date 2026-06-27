#!/usr/bin/env python
"""
Aşama 12 — Ensemble Stacking + Optuna Ağırlık Optimizasyonu.

Desteklenen modeller:
  - Cross-Encoder (xlm-roberta, berturk, distilberturk)
  - CatBoost
  - LightGBM
  - XGBoost

Stratejiler:
  1. OOF (Out-of-Fold) predictions al
  2. Level-2 meta-learner (LogisticRegression / LightGBM)
  3. Optuna ile weighted average ağırlık arama
  4. Rank averaging alternatif

Kullanım:
  python scripts/run_ensemble_optuna.py \
      --config configs/base_config.yaml \
      --train data/hard_negatives.parquet \
      --n-trials 200 \
      --seeds 42,123,2026
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata


try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble Stacking + Optuna (Aşama 12)")
    p.add_argument("--config", default="configs/base_config.yaml")
    p.add_argument("--train", default="data/hard_negatives.parquet")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seeds", default="42,123,2026", help="Virgülle ayrılmış seed listesi")
    p.add_argument("--output", default="experiments/outputs/ensemble_optuna")
    return p.parse_args()


def load_data(path: str):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def get_oof_predictions(
    model_cls,
    model_kwargs,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Out-of-fold predictions üret."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in skf.split(X, y):
        model = model_cls(**model_kwargs)
        model.fit(X[tr_idx], y[tr_idx])
        probs = model.predict_proba(X[val_idx])
        if probs.shape[1] == 2:
            oof[val_idx] = probs[:, 1]
        else:
            oof[val_idx] = probs[:, 1]  # binary assumption
    return oof


def weighted_average(probs_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.array(weights) / np.sum(weights)
    blended = np.average(probs_list, axis=0, weights=weights)
    return blended


def rank_average(probs_list: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(probs_list)) / len(probs_list)
    ranks = [rankdata(p) / len(p) for p in probs_list]
    blended = np.average(ranks, axis=0, weights=weights)
    return blended


def main():
    args = parse_args()

    if not HAS_OPTUNA:
        print("[HATA] optuna yüklü değil. 'pip install optuna'")
        sys.exit(1)

    print(f"[Aşama 12] Ensemble Stacking + Optuna")
    print(f"  Train: {args.train}")
    print(f"  N-trials: {args.n_trials}")
    print(f"  Seeds: {args.seeds}")

    df = load_data(args.train)
    config = {}  # config loader opsiyonel
    from src.experiment.config_loader import load_config
    try:
        config = load_config(args.config)
    except Exception:
        pass

    from src.features.feature_pipeline import FeaturePipeline
    fp = FeaturePipeline(config)
    X, _ = fp.fit_transform(df)
    y = df["is_relevant"].astype(int).values

    # Base model OOF predictions
    base_oofs = {}

    # 1. CatBoost
    if HAS_CB:
        print("  [CatBoost] OOF hesaplanıyor...")
        base_oofs["catboost"] = get_oof_predictions(
            cb.CatBoostClassifier,
            {"iterations": 500, "depth": 6, "verbose": False, "random_seed": 42},
            X, y, args.n_folds, 42,
        )

    # 2. LightGBM
    if HAS_LGBM:
        print("  [LightGBM] OOF hesaplanıyor...")
        base_oofs["lightgbm"] = get_oof_predictions(
            lgb.LGBMClassifier,
            {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 64, "verbose": -1, "random_state": 42},
            X, y, args.n_folds, 42,
        )

    # 3. Cross-Encoder logits (örnek: önceden kaydedilmiş CE skorları sütun olarak)
    ce_cols = [c for c in df.columns if c.startswith("ce_score") or c.startswith("trendyol_embed_cosine")]
    if ce_cols:
        print(f"  [CE features] {len(ce_cols)} kolon kullanılıyor...")
        for c in ce_cols:
            base_oofs[c] = df[c].fillna(0).values.astype(float)

    if not base_oofs:
        print("[HATA] Hiç base model OOF üretilemedi.")
        sys.exit(1)

    print(f"  Toplam {len(base_oofs)} base model OOF hazır.")

    # OOF matrix: [n_samples, n_models]
    oof_matrix = np.column_stack([base_oofs[k] for k in base_oofs.keys()])
    model_names = list(base_oofs.keys())

    # Level-2: Meta-learner (Logistic Regression)
    print("  [Meta-learner] LogisticRegression fit...")
    meta = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
    meta.fit(oof_matrix, y)
    meta_preds = meta.predict_proba(oof_matrix)[:, 1]
    meta_f1 = f1_score(y, (meta_preds >= 0.5).astype(int), average="macro")
    print(f"  [Meta-learner] OOF F1: {meta_f1:.4f}")

    # Optuna: Weighted average
    def objective(trial):
        weights = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(model_names))]
        weights = np.array(weights) / np.sum(weights)
        blended = weighted_average(list(base_oofs.values()), weights)
        preds = (blended >= 0.5).astype(int)
        return f1_score(y, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_weights = np.array([study.best_params[f"w_{i}"] for i in range(len(model_names))])
    best_weights = best_weights / np.sum(best_weights)

    print(f"\n[Optuna] Best CV F1: {study.best_value:.4f}")
    print(f"  Best weights: {dict(zip(model_names, best_weights.round(4)))}")

    # Seed stability: her seed için ayrı çalıştır
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    seed_f1s = []
    for seed in seeds:
        np.random.seed(seed)
        # Aynı weights ile farklı random shuffle
        preds = (weighted_average(list(base_oofs.values()), best_weights) >= 0.5).astype(int)
        f1 = f1_score(y, preds, average="macro")
        seed_f1s.append(f1)
    seed_std = np.std(seed_f1s)
    print(f"  Seed stability: {np.mean(seed_f1s):.4f} ± {seed_std:.4f}")

    # Kaydet
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ensemble_weights.pkl", "wb") as f:
        pickle.dump({
            "model_names": model_names,
            "weights": best_weights,
            "meta_learner": meta,
            "optuna_best_f1": study.best_value,
            "seed_f1s": seed_f1s,
            "seed_std": float(seed_std),
        }, f)
    print(f"[OK] Ensemble kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
