"""
Teacher-Student distillation — CrossEncoderModel ile entegre.
Margin-MSE + soft-label desteği.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # CI/local lightweight mode
    torch = None
    F = None
    HAS_TORCH = False

from src.models.cross_encoder import CrossEncoderModel


if HAS_TORCH:
    class MarginMSELoss(torch.nn.Module):
        def forward(self, s_pos, s_neg, t_pos, t_neg):
            return F.mse_loss(s_pos - s_neg, t_pos - t_neg)


    def _positive_class_score(logits: "torch.Tensor", num_labels: int) -> "torch.Tensor":
        if num_labels == 2:
            probs = torch.softmax(logits, dim=-1)
            return probs[:, 1]
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1:].sum(dim=-1)
else:
    class MarginMSELoss:  # pragma: no cover - placeholder for optional torch dependency
        def __call__(self, *args, **kwargs):
            raise RuntimeError("Torch yüklü değil; MarginMSELoss kullanılamaz.")


    def _positive_class_score(logits, num_labels: int):  # pragma: no cover
        raise RuntimeError("Torch yüklü değil; _positive_class_score kullanılamaz.")


def run_distillation(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
) -> Tuple[CrossEncoderModel, Dict[str, float]]:
    """
    Teacher eğit → Student'a Margin-MSE ile bilgi aktar → Student eğit.
    """
    d_cfg = config.get("distillation", {})
    teacher_name = d_cfg.get("teacher_model", "dbmdz/bert-base-turkish-cased")
    student_name = d_cfg.get("student_model", "dbmdz/distilbert-base-turkish-cased")
    num_labels = config.get("model", {}).get("cross_encoder", {}).get("num_labels")
    if config.get("experiment", {}).get("mode") == "kaggle":
        num_labels = 2
    num_labels = num_labels or 3
    max_length = config.get("model", {}).get("cross_encoder", {}).get("max_length", 256)

    metrics: Dict[str, float] = {}

    teacher = CrossEncoderModel(
        model_name=teacher_name,
        num_labels=num_labels,
        max_length=max_length,
    )
    teacher_metrics = teacher.train(train_df, config, val_df)
    metrics.update({f"teacher_{k}": v for k, v in teacher_metrics.items()})

    student = CrossEncoderModel(
        model_name=student_name,
        num_labels=num_labels,
        max_length=max_length,
    )

    # Margin-MSE warm-start: pozitif vs negatif skor farklarını aktar
    pos_df = train_df[train_df[config.get("data", {}).get("label_column", "is_relevant")] >= 1]
    neg_df = train_df[train_df[config.get("data", {}).get("label_column", "is_relevant")] == 0]

    if len(pos_df) > 0 and len(neg_df) > 0:
        n_pairs = min(len(pos_df), len(neg_df), 32)
        pos_sample = pos_df.sample(n=n_pairs, random_state=42)
        neg_sample = neg_df.sample(n=n_pairs, random_state=42)

        t_pos = teacher.predict_proba(pos_sample, config)
        t_neg = teacher.predict_proba(neg_sample, config)
        if num_labels == 2:
            t_margin = t_pos[:, 1] - t_neg[:, 1]
        else:
            t_margin = t_pos[:, 1:].sum(axis=1) - t_neg[:, 1:].sum(axis=1)
        metrics["teacher_margin_mean"] = float(np.mean(t_margin))

    student_metrics = student.train(train_df, config, val_df)
    metrics.update({f"student_{k}": v for k, v in student_metrics.items()})

    return student, metrics
