"""Real HuggingFace transformer fine-tuning backend for pair classification."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src_kaggle.data.schema import SCHEMA
from src_kaggle.models.transformer_checkpointing import write_metrics, write_training_config
from src_kaggle.models.transformer_dataset import PairClassificationDataset, token_length_diagnostics
from src_kaggle.models.transformer_inference import predict_proba_transformer


def _imports():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
    except Exception as e:
        raise RuntimeError("backend=transformers requires torch and transformers. Install dependencies and rerun; no silent fallback is performed.") from e
    return torch, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, set_seed


def compute_binary_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    return {"macro_f1": float(macro), "class0_f1": float(f1[0]), "class1_f1": float(f1[1])}


class TransformerCrossEncoder:
    def __init__(self, config: dict):
        self.config = dict(config)
        self.model = None
        self.tokenizer = None
        self.model_name = self.config.get("model_name", "dbmdz/distilbert-base-turkish-cased")
        self.tokenizer_name = self.config.get("tokenizer_name") or self.model_name
        self.text_format_version = self.config.get("text_format_version", "full_v1")
        self.max_length = int(self.config.get("max_length", 256))

    def load_base(self):
        torch, AutoModel, AutoTokenizer, *_ = _imports()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModel.from_pretrained(self.model_name, num_labels=2, problem_type="single_label_classification")
        return self

    @classmethod
    def load(cls, checkpoint_dir: str | Path, config: dict | None = None) -> "TransformerCrossEncoder":
        torch, AutoModel, AutoTokenizer, *_ = _imports()
        cfg = dict(config or {})
        obj = cls(cfg)
        obj.tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint_dir) / "tokenizer")
        obj.model = AutoModel.from_pretrained(Path(checkpoint_dir) / "model")
        return obj

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str | Path) -> dict:
        torch, AutoModel, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, set_seed = _imports()
        if not self.config.get("allow_cpu", False) and not torch.cuda.is_available():
            raise RuntimeError("backend=transformers requested but CUDA is not available. Set allow_cpu=true explicitly for tiny/debug runs.")
        set_seed(int(self.config.get("seed", 42)))
        self.load_base()
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        token_stats = token_length_diagnostics(val_df, self.tokenizer, self.text_format_version, self.max_length).to_dict()
        train_ds = PairClassificationDataset(train_df, self.tokenizer, self.text_format_version, self.max_length)
        val_ds = PairClassificationDataset(val_df, self.tokenizer, self.text_format_version, self.max_length)
        args = TrainingArguments(
            output_dir=str(output_dir / "hf_checkpoints"),
            learning_rate=float(self.config.get("learning_rate", 2e-5)),
            per_device_train_batch_size=int(self.config.get("batch_size", 16)),
            per_device_eval_batch_size=int(self.config.get("eval_batch_size", 32)),
            num_train_epochs=float(self.config.get("num_train_epochs", self.config.get("epochs", 2))),
            weight_decay=float(self.config.get("weight_decay", 0.01)),
            warmup_ratio=float(self.config.get("warmup_ratio", 0.1)),
            gradient_accumulation_steps=int(self.config.get("gradient_accumulation_steps", 1)),
            fp16=bool(self.config.get("fp16", torch.cuda.is_available())),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=bool(self.config.get("load_best_model_at_end", True)),
            metric_for_best_model=self.config.get("metric_for_best_model", "macro_f1"),
            greater_is_better=True,
            save_total_limit=int(self.config.get("save_total_limit", 2)),
            logging_steps=int(self.config.get("logging_steps", 50)),
            report_to=[],
            seed=int(self.config.get("seed", 42)),
        )
        callbacks = []
        if int(self.config.get("early_stopping_patience", 0)) > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(self.config.get("early_stopping_patience", 2))))
        start = time.time()
        trainer = Trainer(model=self.model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_binary_metrics, callbacks=callbacks)
        trainer.train()
        metrics = trainer.evaluate()
        self.model = trainer.model
        model_dir = output_dir / "model"; tok_dir = output_dir / "tokenizer"
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(tok_dir)
        metrics = {**metrics, "training_time_sec": float(time.time() - start), "token_stats": token_stats, "best_checkpoint": str(trainer.state.best_model_checkpoint), "model_name": self.model_name, "tokenizer_name": self.tokenizer_name, "text_format_version": self.text_format_version, "max_length": self.max_length}
        write_training_config(output_dir / "training_config.json", self.config)
        write_metrics(output_dir / "metrics.json", metrics)
        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("TransformerCrossEncoder is not loaded/fitted")
        return predict_proba_transformer(self.model, self.tokenizer, df, text_format_version=self.text_format_version, max_length=self.max_length, batch_size=int(self.config.get("eval_batch_size", 32)), device=self.config.get("device"))
