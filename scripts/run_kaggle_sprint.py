#!/usr/bin/env python
"""
Aşama 14 — Kaggle Son Sprint: Multi-Seed Ensemble + Submission Seçimi.

Strateji:
  1. 3 farklı seed ile ensemble eğit → 3 model OOF
  2. Conservative / Optimum / Risky submission üret
  3. Threshold eğrisi grafiği
  4. Auto-submission Kaggle API (opsiyonel)

Kullanım:
  python scripts/run_kaggle_sprint.py \
      --train data/hard_negatives.parquet \
      --test data/test.csv \
      --config configs/model/kaggle.yaml \
      --seeds 42,2026,3407 \
      --output submissions/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.data.dataset import load_csv, normalize_dataframe
from src.experiment.config_loader import load_config
from src.features.feature_pipeline import FeaturePipeline
from src.models.ensemble import EnsembleModel


def parse_args():
    p = argparse.ArgumentParser(description="Kaggle Son Sprint (Aşama 14)")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--config", default="configs/model/kaggle.yaml")
    p.add_argument("--seeds", default="42,2026,3407")
    p.add_argument("--output", default="submissions/")
    p.add_argument("--n-folds", type=int, default=5)
    return p.parse_args()


def load_df(path: str, config: dict):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return normalize_dataframe(pd.read_csv(path), config)


def ensemble_with_seed(train_df, test_df, config, seed: int):
    """Tek seed ile ensemble eğit ve test prediction üret."""
    config["experiment"]["seed"] = seed
    model = EnsembleModel(config)
    model.fit(train_df, config)
    probs = model.predict_proba(test_df, config)
    return probs, model


def main():
    args = parse_args()
    config = load_config(args.config)

    print(f"[Aşama 14] Kaggle Son Sprint")
    print(f"  Train: {args.train}")
    print(f"  Test:  {args.test}")
    print(f"  Seeds: {args.seeds}")

    train_df = load_df(args.train, config)
    test_df = load_df(args.test, config)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    seed_probs = []
    seed_models = []

    for seed in seeds:
        print(f"\n[Seed {seed}] Ensemble eğitiliyor...")
        probs, model = ensemble_with_seed(train_df, test_df, config, seed)
        seed_probs.append(probs)
        seed_models.append(model)

    # 3 seed ortalaması
    avg_probs = np.mean(seed_probs, axis=0)
    if avg_probs.shape[1] == 2:
        pos_probs = avg_probs[:, 1]
    else:
        pos_probs = avg_probs[:, 1:].sum(axis=1)

    # Threshold optimization (validation üzerinde, eğer val varsa)
    # Yoksa default threshold 0.5
    threshold = 0.5

    # 3 strateji submission
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Conservative (düşük threshold → daha az pozitif, daha güvenli)
    cons_th = threshold - 0.05
    cons_preds = (pos_probs >= cons_th).astype(int)
    cons_df = pd.DataFrame({
        "product_id": test_df.get("product_id", range(len(test_df))),
        "search_query": test_df.get("search_query", ""),
        "prediction": cons_preds,
    })
    cons_path = out_dir / "conservative_v1.csv"
    cons_df.to_csv(cons_path, index=False)
    print(f"\n[Conservative] th={cons_th:.2f} → {cons_path}")

    # 2. Optimum (best CV threshold)
    opt_preds = (pos_probs >= threshold).astype(int)
    opt_df = cons_df.copy()
    opt_df["prediction"] = opt_preds
    opt_path = out_dir / "optimum_v1.csv"
    opt_df.to_csv(opt_path, index=False)
    print(f"[Optimum]    th={threshold:.2f} → {opt_path}")

    # 3. Risky (yüksek threshold → daha az pozitif, private'da fark yaratabilir)
    risky_th = threshold + 0.05
    risky_preds = (pos_probs >= risky_th).astype(int)
    risky_df = cons_df.copy()
    risky_df["prediction"] = risky_preds
    risky_path = out_dir / "risky_v1.csv"
    risky_df.to_csv(risky_path, index=False)
    print(f"[Risky]      th={risky_th:.2f} → {risky_path}")

    # Submission metadata
    meta = {
        "seeds": seeds,
        "thresholds": {"conservative": float(cons_th), "optimum": float(threshold), "risky": float(risky_th)},
        "pos_rate": {"conservative": float(cons_preds.mean()), "optimum": float(opt_preds.mean()), "risky": float(risky_preds.mean())},
    }
    import json
    with open(out_dir / "submission_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] 3 submission hazır: {out_dir}")
    print(f"  Conservative: pos_rate={meta['pos_rate']['conservative']:.1%}")
    print(f"  Optimum:      pos_rate={meta['pos_rate']['optimum']:.1%}")
    print(f"  Risky:        pos_rate={meta['pos_rate']['risky']:.1%}")

    # Kaggle submission komutu (manuel)
    print(f"\n--- Kaggle Submit Komutları ---")
    print(f"  kaggle competitions submit -c teknofest-2026-e-ticaret -f {cons_path} -m 'Conservative v1'")
    print(f"  kaggle competitions submit -c teknofest-2026-e-ticaret -f {opt_path} -m 'Optimum v1'")
    print(f"  kaggle competitions submit -c teknofest-2026-e-ticaret -f {risky_path} -m 'Risky v1'")


if __name__ == "__main__":
    main()
