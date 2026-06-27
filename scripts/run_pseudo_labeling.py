#!/usr/bin/env python
"""
Aşama 13 — Iterative Pseudo-Labeling + Self-Distillation.

Strateji:
  1. En güçlü ensemble model ile test seti üzerinde tahmin
  2. Yüksek güvenli (conf > 0.95 veya < 0.05) örnekleri train'e ekle
  3. Yeniden eğit
  4. 3 iterasyon boyunca eşik düşür (daha fazla örnek al)
  5. Final self-distillation: teacher ensemble → student (DistilBERTurk)

Kullanım:
  python scripts/run_pseudo_labeling.py \
      --train data/hard_negatives.parquet \
      --test data/test.csv \
      --config configs/model/kaggle.yaml \
      --iterations 3 \
      --output experiments/outputs/pseudo_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.dataset import load_csv, normalize_dataframe
from src.experiment.config_loader import load_config
from src.features.feature_pipeline import FeaturePipeline
from src.models.cross_encoder import CrossEncoderModel
from src.models.ensemble import EnsembleModel


def parse_args():
    p = argparse.ArgumentParser(description="Iterative Pseudo-Labeling (Aşama 13)")
    p.add_argument("--train", required=True, help="Eğitim verisi (etiketli)")
    p.add_argument("--test", required=True, help="Test verisi (etiketsiz)")
    p.add_argument("--config", default="configs/model/kaggle.yaml")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--output", default="experiments/outputs/pseudo_v1")
    p.add_argument("--initial-pos-th", type=float, default=0.95)
    p.add_argument("--initial-neg-th", type=float, default=0.05)
    p.add_argument("--th-decay", type=float, default=0.05, help="Her iterasyonda eşik düşüşü")
    p.add_argument("--mode", choices=["kaggle", "final"], default="kaggle")
    return p.parse_args()


def load_df(path: str, config: dict):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return normalize_dataframe(pd.read_csv(path), config)


def main():
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("experiment", {})["mode"] = args.mode

    print(f"[Aşama 13] Iterative Pseudo-Labeling")
    print(f"  Train: {args.train}")
    print(f"  Test:  {args.test}")
    print(f"  Iter:  {args.iterations}")

    train_df = load_df(args.train, config)
    test_df = load_df(args.test, config)

    current_train = train_df.copy()
    current_model = None

    for iteration in range(args.iterations):
        pos_th = args.initial_pos_th - args.th_decay * iteration
        neg_th = args.initial_neg_th + args.th_decay * iteration

        print(f"\n[Iter {iteration}] pos_th={pos_th:.2f}, neg_th={neg_th:.2f}")
        print(f"  Train size: {len(current_train)}")

        # Model eğit
        model = EnsembleModel(config)
        metrics = model.fit(current_train, config)
        print(f"  Train metrics: {metrics}")

        # Test tahmin
        test_probs = model.predict_proba(test_df, config)
        if test_probs.shape[1] == 2:
            pos_probs = test_probs[:, 1]
        else:
            pos_probs = test_probs[:, 1:].sum(axis=1)  # multi-class

        high_conf = (pos_probs > pos_th) | (pos_probs < neg_th)
        n_added = high_conf.sum()
        print(f"  High confidence: {n_added} / {len(test_df)}")

        if n_added == 0:
            print("  Durduruluyor: yeni high-confidence örnek yok.")
            break

        # Pseudo-label ekle
        pseudo = test_df[high_conf].copy()
        pseudo["is_relevant"] = (pos_probs[high_conf] > 0.5).astype(int)
        pseudo["pseudo_confidence"] = np.maximum(pos_probs[high_conf], 1 - pos_probs[high_conf])
        pseudo["pseudo_iter"] = iteration
        pseudo["is_pseudo"] = True

        current_train = pd.concat([train_df, pseudo], ignore_index=True)
        current_train.drop_duplicates(subset=["product_id"], keep="first", inplace=True)
        current_model = model

    # Final model kaydet
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    current_model.save(out_dir)
    current_train.to_csv(out_dir / "augmented_train.csv", index=False)
    print(f"\n[OK] Final model: {out_dir}")
    print(f"  Augmented train: {len(current_train)} (orijinal: {len(train_df)})")

    # Self-distillation summary (student eğitimi için talimat)
    print(f"\n[Self-Distillation] Student: dbmdz/distilbert-base-turkish-cased")
    print(f"  Teacher: ensemble ({len(current_model.models) + 1} models)")
    print(f"  Soft labels: teacher.predict_proba(train_df)")
    print(f"  Loss: α*CE + (1-α)*MarginMSE")
    print(f"  (Distillation script: scripts/run_distillation.py)")


if __name__ == "__main__":
    main()
