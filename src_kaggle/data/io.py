"""I/O helpers for Kaggle War Mode with strict data contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src_kaggle.data.contracts import TableKind, validate_dataframe
from src_kaggle.data.schema import SCHEMA


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_contract_table(path: str | Path, kind: TableKind | str, *, positive_only: bool = False) -> pd.DataFrame:
    return validate_dataframe(read_table(path), kind, positive_only=positive_only)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def merge_terms(pairs: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    return pairs.merge(terms[[SCHEMA.term_id, SCHEMA.query]], on=SCHEMA.term_id, how="left", validate="many_to_one")


def merge_items(pairs: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    cols = [
        SCHEMA.item_id,
        SCHEMA.title,
        SCHEMA.category,
        SCHEMA.brand,
        SCHEMA.gender,
        SCHEMA.age_group,
        SCHEMA.attributes,
    ]
    return pairs.merge(items[cols], on=SCHEMA.item_id, how="left", validate="many_to_one")
