"""OOF/public correlation metrics with small-sample safeguards."""
from __future__ import annotations

from typing import Any
import pandas as pd


def safe_corr(df: pd.DataFrame, x: str = "OOF macro-F1", y: str = "public_lb_score") -> dict[str, Any]:
    part = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    n = int(len(part))
    if n < 3:
        return {"n": n, "pearson": None, "spearman": None, "warning": "too_few_public_lb_points_for_reliable_correlation"}
    return {"n": n, "pearson": float(part[x].corr(part[y], method="pearson")), "spearman": float(part[x].corr(part[y], method="spearman")), "warning": "interpret_with_caution_public_subset_is_not_private"}


def sign_agreement(df: pd.DataFrame) -> dict[str, Any]:
    if "oof_delta_vs_prev" not in df.columns or "public_delta_vs_prev" not in df.columns:
        tmp = df.copy()
        tmp["oof_delta_vs_prev"] = pd.to_numeric(tmp.get("OOF macro-F1"), errors="coerce").diff()
        tmp["public_delta_vs_prev"] = pd.to_numeric(tmp.get("public_lb_score"), errors="coerce").diff()
    else:
        tmp = df
    part = tmp[["oof_delta_vs_prev", "public_delta_vs_prev"]].apply(pd.to_numeric, errors="coerce").dropna()
    if part.empty:
        return {"n": 0, "sign_agreement_rate": None}
    agree = (part["oof_delta_vs_prev"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0) == part["public_delta_vs_prev"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).mean()
    return {"n": int(len(part)), "sign_agreement_rate": float(agree)}
