"""Pseudo-labeling — teacher model ile soft/hard etiket üretimi."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class PseudoLabeler:
    def __init__(self, teacher_model, batch_size: int = 64):
        self.teacher = teacher_model
        self.batch_size = batch_size

    def generate_labels(
        self, unlabelled_df: pd.DataFrame, config: Dict[str, Any] | None = None
    ) -> pd.DataFrame:
        config = config or {}
        q_col = config.get("data", {}).get("query_column", "search_query")
        t_col = config.get("data", {}).get("product_text_column", "product_name")
        label_col = config.get("data", {}).get("label_column", "is_relevant")

        if q_col not in unlabelled_df.columns or t_col not in unlabelled_df.columns:
            raise ValueError(f"DataFrame must contain '{q_col}' and '{t_col}'.")

        df = unlabelled_df.copy()
        probs = self.teacher.predict_proba(df, config)
        df["pseudo_confidence"] = probs.max(axis=1)
        df[label_col] = np.argmax(probs, axis=1)
        df["pseudo_probs"] = list(probs)
        return df
