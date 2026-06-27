"""Macro-F1 için optimal eşik değeri arama (global + kategori bazlı)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def search_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: Optional[List[float]] = None,
    num_classes: int = 3,
) -> Tuple[float, float, Dict[float, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores)

    if y_scores.ndim == 2:
        if num_classes == 2:
            scores = y_scores[:, 1]
        else:
            preds = np.argmax(y_scores, axis=1)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            return 0.5, float(f1), {0.5: float(f1)}
    else:
        scores = y_scores

    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 95, 5)]

    results = {}
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, average="binary", zero_division=0)
        results[t] = float(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1, results


def _apply_category_thresholds(
    y_probs: np.ndarray,
    df: pd.DataFrame,
    cat_thresholds: Dict[str, Dict[str, float]],
    category_col: str,
    num_classes: int,
) -> np.ndarray:
    preds = np.argmax(y_probs, axis=1)
    if num_classes != 2 or category_col not in df.columns:
        return preds

    for i, (_, row) in enumerate(df.iterrows()):
        cat = str(row.get(category_col, "default"))
        th_map = cat_thresholds.get(cat, cat_thresholds.get("default", {}))
        pos_th = th_map.get("positive", 0.5)
        if y_probs[i, 1] >= pos_th:
            preds[i] = 1
        else:
            preds[i] = 0
    return preds


def search_category_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    df: pd.DataFrame,
    config: Dict[str, Any],
    category_col: str = "category",
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Kategori bazlı binary eşik araması."""
    thresholds = config.get("threshold", {}).get(
        "search_range", [0.3, 0.4, 0.5, 0.6, 0.7]
    )
    num_classes = y_probs.shape[1]
    if num_classes != 2 or category_col not in df.columns:
        return {}, 0.0

    y_true = np.asarray(y_true).astype(int)
    cat_thresholds: Dict[str, Dict[str, float]] = {}
    all_preds = []

    categories = df[category_col].fillna("default").astype(str).unique()
    for cat in categories:
        mask = df[category_col].fillna("default").astype(str) == cat
        if mask.sum() < 3:
            cat_thresholds[cat] = {"positive": 0.5}
            continue
        best_t, best_f1, _ = search_threshold(
            y_true[mask], y_probs[mask], thresholds, num_classes=2
        )
        cat_thresholds[cat] = {"positive": best_t, "f1": best_f1}

    cat_thresholds.setdefault("default", {"positive": 0.5})
    preds = _apply_category_thresholds(y_probs, df, cat_thresholds, category_col, num_classes)
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    return cat_thresholds, macro_f1


def search_multiclass_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    config: Dict[str, Any],
    val_df: Optional[pd.DataFrame] = None,
    category_col: str = "category",
) -> Tuple[Dict[str, float], float, Dict[str, Dict[str, float]]]:
    y_true = np.asarray(y_true).astype(int)
    n_classes = y_probs.shape[1]
    thresholds = config.get("threshold", {}).get(
        "search_range", [0.3, 0.4, 0.5, 0.6, 0.7]
    )

    best_thresholds = {}
    for c in range(n_classes):
        binary_true = (y_true == c).astype(int)
        best_t, _, _ = search_threshold(binary_true, y_probs[:, c], thresholds, n_classes)
        best_thresholds[f"class_{c}"] = best_t

    preds = np.argmax(y_probs, axis=1)
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

    cat_thresholds: Dict[str, Dict[str, float]] = {}
    if val_df is not None and config.get("threshold", {}).get("category_specific", True):
        cat_thresholds, cat_f1 = search_category_thresholds(
            y_true, y_probs, val_df, config, category_col
        )
        if cat_f1 > 0:
            macro_f1 = max(macro_f1, cat_f1)

    return best_thresholds, macro_f1, cat_thresholds
