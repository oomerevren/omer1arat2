"""OOF ensemble/blend optimizer for final submission families."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src_kaggle.data.schema import SCHEMA


def _load_oof(path: str | Path) -> pd.DataFrame | None:
    p = Path(str(path))
    if not str(path) or not p.exists():
        return None
    df = pd.read_csv(p)
    if "proba" not in df.columns or SCHEMA.label not in df.columns:
        return None
    if SCHEMA.id not in df.columns:
        df[SCHEMA.id] = range(len(df))
    return df[[SCHEMA.id, SCHEMA.label, "proba"]].copy()


def _best_threshold(y, p, start=0.05, end=0.95, step=0.01):
    best_t, best = 0.5, {"macro_f1": -1, "class0_f1": 0, "class1_f1": 0}
    for t in np.arange(start, end + 1e-9, step):
        pred = (p >= t).astype(int)
        c0 = f1_score(y, pred, pos_label=0, zero_division=0)
        c1 = f1_score(y, pred, pos_label=1, zero_division=0)
        macro = (c0 + c1) / 2
        if macro > best["macro_f1"]:
            best_t, best = float(t), {"macro_f1": float(macro), "class0_f1": float(c0), "class1_f1": float(c1)}
    return best_t, best


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks / max(1, len(x)-1)


def prediction_correlation(oofs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not oofs:
        return pd.DataFrame()
    base = None
    data = {}
    for name, df in oofs.items():
        if base is None:
            base = df[[SCHEMA.id]].copy()
        merged = base.merge(df[[SCHEMA.id, "proba"]], on=SCHEMA.id, how="left")
        data[name] = merged["proba"].values
    return pd.DataFrame(data).corr()


def optimize_blends(candidate_pool: pd.DataFrame, max_models: int = 3) -> tuple[pd.DataFrame, dict[str, Any]]:
    ready = candidate_pool[candidate_pool.get("selection_status", "").eq("ready")].copy() if not candidate_pool.empty else pd.DataFrame()
    oofs: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    for _, r in ready.iterrows():
        df = _load_oof(r.get("oof_path", ""))
        if df is not None:
            cid = str(r["candidate_id"])
            oofs[cid] = df
            meta[cid] = r.to_dict()
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if len(oofs) == 0:
        report = {
            "tested_blends": [], "best_weighted_blends": [], "best_rank_blends": [],
            "rejected_blends": [{"reason": "no_ready_oof_candidates", "detail": "Official OOF candidate files are required."}],
            "prediction_correlation": {}, "family_mapping": {},
        }
        return pd.DataFrame(), report

    names = list(oofs)
    # align by first candidate ids
    base = oofs[names[0]][[SCHEMA.id, SCHEMA.label]].copy()
    aligned = {}
    for n in names:
        aligned[n] = base.merge(oofs[n][[SCHEMA.id, "proba"]], on=SCHEMA.id, how="left")["proba"].fillna(0).values
    y = base[SCHEMA.label].astype(int).values

    for k in range(1, min(max_models, len(names)) + 1):
        for combo in combinations(names, k):
            # OOF-weighted safe default, class0 boosted for defensive variants downstream.
            scores = np.array([max(1e-6, float(meta[n].get("OOF macro-F1", 0) or 0)) for n in combo])
            weights = scores / scores.sum()
            p = sum(w * aligned[n] for w, n in zip(weights, combo))
            t, m = _best_threshold(y, p)
            corr = float(pd.DataFrame({n: aligned[n] for n in combo}).corr().where(~np.eye(len(combo), dtype=bool)).stack().mean()) if len(combo) > 1 else 1.0
            row = {"blend_id": "wa__" + "__".join(combo), "blend_method": "weighted_average", "models": "|".join(combo), "weights": "|".join(f"{n}:{w:.4f}" for n, w in zip(combo, weights)), "threshold": t, **m, "prediction_correlation_mean": corr, "n_models": k}
            rows.append(row)
            if corr > 0.985 and k > 1:
                rejected.append({**row, "reject_reason": "too_high_prediction_correlation_redundant"})
            # rank blend
            rp = sum(w * _rankdata(aligned[n]) for w, n in zip(weights, combo))
            rt, rm = _best_threshold(y, rp)
            row2 = {"blend_id": "rank__" + "__".join(combo), "blend_method": "rank_average", "models": "|".join(combo), "weights": "|".join(f"{n}:{w:.4f}" for n, w in zip(combo, weights)), "threshold": rt, **rm, "prediction_correlation_mean": corr, "n_models": k}
            rows.append(row2)
    table = pd.DataFrame(rows).sort_values(["macro_f1", "class0_f1"], ascending=False).reset_index(drop=True)
    corr_df = prediction_correlation(oofs)
    report = {
        "tested_blends": table.to_dict("records"),
        "best_weighted_blends": table[table["blend_method"].eq("weighted_average")].head(10).to_dict("records"),
        "best_rank_blends": table[table["blend_method"].eq("rank_average")].head(10).to_dict("records"),
        "rejected_blends": rejected,
        "prediction_correlation": corr_df.to_dict() if not corr_df.empty else {},
        "family_mapping": {cid: meta[cid].get("model_family", "unknown") for cid in meta},
    }
    return table, report


def write_blend_comparison(table: pd.DataFrame, report: dict[str, Any], out_json="reports/final/final_blend_comparison.json", out_md="reports/final/final_blend_comparison.md") -> None:
    pj = Path(out_json); pm = Path(out_md); pj.parent.mkdir(parents=True, exist_ok=True)
    pj.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines = ["# Final Blend Comparison", "", "## Best weighted blends", ""]
    if table.empty:
        lines += ["No executable blends: ready OOF candidate files are missing.", ""]
    else:
        lines.append(table[table["blend_method"].eq("weighted_average")].head(10).to_csv(index=False))
        lines += ["", "## Best rank blends", "", table[table["blend_method"].eq("rank_average")].head(10).to_csv(index=False)]
    lines += ["", "## Rejected blends", ""]
    rej = pd.DataFrame(report.get("rejected_blends", []))
    lines.append(rej.to_csv(index=False) if not rej.empty else "No rejected blend or no OOF data yet.")
    pm.write_text("\n".join(lines) + "\n", encoding="utf-8")
