#!/usr/bin/env python
"""Train the canonical Kaggle War Mode baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.data.io import read_table
from src_kaggle.training.pipeline import cross_validate_baseline, train_full_baseline
from src_kaggle.utils.config import load_kaggle_config
from src_kaggle.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train Kaggle War Mode baseline")
    p.add_argument("--config", default="configs/kaggle/war_mode.yaml")
    p.add_argument("--train-pairs", default=None)
    p.add_argument("--model-out", default=None)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_kaggle_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["training"]["seed"])
    folds = args.folds if args.folds is not None else int(cfg["training"]["folds"])
    train_pairs = args.train_pairs or cfg["paths"].get("train_model_input", cfg["paths"].get("train_merged"))
    model_out = args.model_out or cfg["paths"]["model"]
    metrics_out = args.metrics_out or cfg["paths"]["metrics"]

    seed_everything(seed)
    df = read_table(train_pairs)
    result = cross_validate_baseline(df, n_splits=folds, seed=seed)
    model = train_full_baseline(df, seed=seed)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    metrics = {
        "mode": "kaggle_war_mode",
        "num_labels": 2,
        "metric": "macro_f1",
        "macro_f1": result.macro_f1,
        "threshold": result.threshold,
        "fold_scores": result.fold_scores,
        "model": cfg["model"]["name"],
    }
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] model: {model_out}")
    print(f"[OK] metrics: {metrics_out}")
    print(f"macro_f1={result.macro_f1:.6f} threshold={result.threshold:.3f}")


if __name__ == "__main__":
    main()
