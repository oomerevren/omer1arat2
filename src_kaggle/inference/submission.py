"""Submission generation helpers for Kaggle War Mode."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src_kaggle.data.schema import SCHEMA


def make_submission_frame(pairs: pd.DataFrame, proba, threshold: float = 0.5) -> pd.DataFrame:
    """Return strict Kaggle submission columns: id,prediction."""
    out = pd.DataFrame()
    if SCHEMA.id in pairs.columns:
        out[SCHEMA.id] = pairs[SCHEMA.id].values
    else:
        # Fallback id only when official test does not provide one. This is
        # explicit and local to Kaggle; no product_id/search_query leakage.
        out[SCHEMA.id] = np.arange(len(pairs))
    out[SCHEMA.prediction] = (proba >= threshold).astype(int)
    return out[[SCHEMA.id, SCHEMA.prediction]]
