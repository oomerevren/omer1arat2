#!/usr/bin/env python
"""
Aşama 9 — Cross-Encoder Fine-Tuning (xlm-roberta-base / berturk / distilberturk).

Kullanım:
  python scripts/train_cross_encoder.py \
      --model FacebookAI/xlm-roberta-base \
      --train data/hard_negatives.parquet \
      --val data/val.parquet \
      --output models/ce_xlm_r_v1

Desteklenen modeller:
  - FacebookAI/xlm-roberta-base (multilingual, Türkçe + İngilizce)
  - dbmdz/bert-base-turkish-cased (Türkçe odaklı)
  - dbmdz/distilbert-base-turkish-cased (hızlı, hafif)
  - BAAI/bge-m3 (multilingual, SOTA)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


try:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    HAS_ST = True
except ImportError:
    HAS_ST = False


def parse_args():
    p = argparse.ArgumentParser(description="Cross-Encoder fine-tuning (Aşama 9)")
    p.add_argument("--model", default="FacebookAI/xlm-roberta-base", help="Pretrained model")
    p.add_argument("--train", default="data/hard_negatives.parquet", help="Train data path")
    p.add_argument("--val", default=None, help="Validation data path (opsiyonel)")
    p.add_argument("--output", default="experiments/outputs/ce_xlm_r_v1", help="Output dir")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--bf16", action="store_true", help="bfloat16 training (Ampere+ GPU)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(path: str, query_col: str = "search_query", product_col: str = "product_name", label_col: str = "is_relevant"):
    """Parquet veya CSV yükler."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # Sentence-transformers format: (query, product, label) listesi
    pairs = []
    for _, row in df.iterrows():
        q = str(row.get(query_col, ""))
        p = str(row.get(product_col, ""))
        label = int(row.get(label_col, 0))
        pairs.append([q, p, float(label)])
    return pairs


def compute_metrics(eval_pred):
    """Macro-F1 evaluation."""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    # CE predict: logits, apply threshold 0.5
    preds = (predictions >= 0.5).astype(int)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"eval_f1_macro": float(f1)}


def main():
    args = parse_args()

    if not HAS_ST:
        print("[HATA] sentence-transformers yüklü değil. 'pip install sentence-transformers'")
        sys.exit(1)

    print(f"[Aşama 9] Cross-Encoder fine-tuning: {args.model}")
    print(f"  Train: {args.train}")
    print(f"  Val:   {args.val}")
    print(f"  Output: {args.output}")

    # Veri yükle
    train_pairs = load_data(args.train)
    print(f"  Train pairs: {len(train_pairs)}")

    val_pairs = None
    if args.val and Path(args.val).exists():
        val_pairs = load_data(args.val)
        print(f"  Val pairs: {len(val_pairs)}")

    # Model
    model = CrossEncoder(
        args.model,
        num_labels=1,
        max_length=args.max_len,
        automodel_args={"torch_dtype": "bfloat16"} if args.bf16 else {},
    )

    loss = BinaryCrossEntropyLoss(model)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "ce_train"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=args.bf16,
        eval_strategy="steps" if val_pairs else "no",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=bool(val_pairs),
        metric_for_best_model="eval_f1_macro",
        gradient_accumulation_steps=2,
        seed=args.seed,
        logging_steps=50,
        report_to="none",
    )

    # Dataset
    from datasets import Dataset
    train_ds = Dataset.from_dict({
        "sentence1": [p[0] for p in train_pairs],
        "sentence2": [p[1] for p in train_pairs],
        "label": [p[2] for p in train_pairs],
    })
    eval_ds = None
    if val_pairs:
        eval_ds = Dataset.from_dict({
            "sentence1": [p[0] for p in val_pairs],
            "sentence2": [p[1] for p in val_pairs],
            "label": [p[2] for p in val_pairs],
        })

    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
        compute_metrics=compute_metrics if eval_ds else None,
    )

    trainer.train()
    model.save(str(out_dir))
    print(f"[OK] Model kaydedildi: {out_dir}")

    # ONNX export (opsiyonel)
    try:
        import torch
        from src.inference.quantizer import export_onnx
        onnx_path = str(out_dir / "model.onnx")
        export_onnx(model.model, model.tokenizer, onnx_path, max_length=args.max_len)
        print(f"[OK] ONNX export: {onnx_path}")
    except Exception as e:
        print(f"[INFO] ONNX export atlandı: {e}")


if __name__ == "__main__":
    main()
