"""Batch inference for fine-tuned transformer cross-encoders."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src_kaggle.models.transformer_dataset import build_pair_texts


def predict_proba_transformer(model, tokenizer, df: pd.DataFrame, *, text_format_version: str = "full_v1", max_length: int = 256, batch_size: int = 32, device: str | None = None) -> np.ndarray:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for transformer inference") from e
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    texts = build_pair_texts(df, text_format_version)
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i+batch_size], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs) if probs else np.asarray([], dtype=float)
