"""Ablation registry helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_ablation_master(path: str | Path = "reports/ablation/ablation_master_table.csv") -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def append_ablation_rows(rows: list[dict], path: str | Path = "reports/ablation/ablation_master_table.csv") -> pd.DataFrame:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(p, index=False)
    return df
