"""Strict data contracts for TEKNOFEST 2026 Kaggle War Mode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src_kaggle.data.schema import (
    ITEM_REQUIRED_COLUMNS,
    SAMPLE_SUBMISSION_REQUIRED_COLUMNS,
    SCHEMA,
    SUBMISSION_PAIRS_REQUIRED_COLUMNS,
    TERMS_REQUIRED_COLUMNS,
    TRAINING_PAIRS_REQUIRED_COLUMNS,
)


class TableKind(str, Enum):
    ITEMS = "items"
    TERMS = "terms"
    TRAINING_PAIRS = "training_pairs"
    SUBMISSION_PAIRS = "submission_pairs"
    SAMPLE_SUBMISSION = "sample_submission"
    TRAIN_FEATURES = "train_features"
    TEST_FEATURES = "test_features"


@dataclass(frozen=True)
class TableContract:
    kind: TableKind
    required_columns: tuple[str, ...]
    require_unique_id: bool = False
    require_unique_pair: bool = False
    require_label: bool = False
    require_no_empty_query: bool = False
    require_no_empty_title: bool = False


CONTRACTS: dict[TableKind, TableContract] = {
    TableKind.ITEMS: TableContract(
        kind=TableKind.ITEMS,
        required_columns=ITEM_REQUIRED_COLUMNS,
        require_unique_id=True,
        require_no_empty_title=True,
    ),
    TableKind.TERMS: TableContract(
        kind=TableKind.TERMS,
        required_columns=TERMS_REQUIRED_COLUMNS,
        require_unique_id=True,
        require_no_empty_query=True,
    ),
    TableKind.TRAINING_PAIRS: TableContract(
        kind=TableKind.TRAINING_PAIRS,
        required_columns=TRAINING_PAIRS_REQUIRED_COLUMNS,
        require_unique_id=True,
        require_unique_pair=True,
        require_label=True,
    ),
    TableKind.SUBMISSION_PAIRS: TableContract(
        kind=TableKind.SUBMISSION_PAIRS,
        required_columns=SUBMISSION_PAIRS_REQUIRED_COLUMNS,
        require_unique_id=True,
        require_unique_pair=True,
    ),
    TableKind.SAMPLE_SUBMISSION: TableContract(
        kind=TableKind.SAMPLE_SUBMISSION,
        required_columns=SAMPLE_SUBMISSION_REQUIRED_COLUMNS,
        require_unique_id=True,
    ),
    TableKind.TRAIN_FEATURES: TableContract(
        kind=TableKind.TRAIN_FEATURES,
        required_columns=(
            SCHEMA.id,
            SCHEMA.term_id,
            SCHEMA.item_id,
            SCHEMA.label,
            SCHEMA.query,
            SCHEMA.title,
            SCHEMA.category,
            SCHEMA.brand,
            SCHEMA.gender,
            SCHEMA.age_group,
            SCHEMA.attributes,
            SCHEMA.attribute_dict,
            SCHEMA.attribute_keys,
            SCHEMA.attribute_values,
            SCHEMA.normalized_attribute_text,
            SCHEMA.color_value,
            SCHEMA.material_value,
            SCHEMA.style_value,
            SCHEMA.full_item_text,
        ),
        require_unique_id=True,
        require_unique_pair=True,
        require_label=True,
        require_no_empty_query=True,
        require_no_empty_title=True,
    ),
    TableKind.TEST_FEATURES: TableContract(
        kind=TableKind.TEST_FEATURES,
        required_columns=(
            SCHEMA.id,
            SCHEMA.term_id,
            SCHEMA.item_id,
            SCHEMA.query,
            SCHEMA.title,
            SCHEMA.category,
            SCHEMA.brand,
            SCHEMA.gender,
            SCHEMA.age_group,
            SCHEMA.attributes,
            SCHEMA.attribute_dict,
            SCHEMA.attribute_keys,
            SCHEMA.attribute_values,
            SCHEMA.normalized_attribute_text,
            SCHEMA.color_value,
            SCHEMA.material_value,
            SCHEMA.style_value,
            SCHEMA.full_item_text,
        ),
        require_unique_id=True,
        require_unique_pair=True,
        require_no_empty_query=True,
        require_no_empty_title=True,
    ),
}


class DataContractError(ValueError):
    """Raised when an input table violates the Kaggle data contract."""


def _empty_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _check_missing_columns(df: pd.DataFrame, contract: TableContract, errors: list[str]) -> None:
    missing = [col for col in contract.required_columns if col not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")


def _check_unique_id(df: pd.DataFrame, id_col: str, errors: list[str]) -> None:
    if id_col not in df.columns:
        return
    duplicated = df[df[id_col].duplicated(keep=False)][id_col].head(10).tolist()
    if duplicated:
        errors.append(f"duplicate {id_col} values detected, examples={duplicated}")


def _check_unique_pair(df: pd.DataFrame, errors: list[str]) -> None:
    pair_cols = [SCHEMA.term_id, SCHEMA.item_id]
    if not all(col in df.columns for col in pair_cols):
        return
    duplicated = df[df.duplicated(pair_cols, keep=False)][pair_cols].head(10).to_dict("records")
    if duplicated:
        errors.append(f"duplicate ({SCHEMA.term_id}, {SCHEMA.item_id}) pairs detected, examples={duplicated}")


def _check_label(df: pd.DataFrame, errors: list[str]) -> None:
    if SCHEMA.label not in df.columns:
        return
    labels = set(pd.Series(df[SCHEMA.label]).dropna().astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        errors.append(f"label must be binary 0/1, got={sorted(labels)}")


def _check_positive_only_training_pairs(df: pd.DataFrame, errors: list[str]) -> None:
    if SCHEMA.label not in df.columns:
        return
    labels = set(pd.Series(df[SCHEMA.label]).dropna().astype(int).unique().tolist())
    if labels != {1}:
        errors.append(f"official training_pairs.csv must contain only label=1, got={sorted(labels)}")


def _check_empty_text(df: pd.DataFrame, column: str, errors: list[str]) -> None:
    if column not in df.columns:
        return
    count = int(_empty_mask(df[column]).sum())
    if count:
        errors.append(f"empty {column} values detected: count={count}")


def _normalize_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (SCHEMA.gender, SCHEMA.age_group):
        if col in out.columns:
            out[col] = out[col].fillna("unknown").astype(str).str.strip()
            out.loc[out[col].eq(""), col] = "unknown"
    return out


def validate_dataframe(df: pd.DataFrame, kind: TableKind | str, *, positive_only: bool = False) -> pd.DataFrame:
    """Validate and return a normalized copy of a dataframe.

    `gender` and `age_group` are intentionally tolerant: missing/empty values are
    normalized to `unknown` instead of raising.
    """
    kind = TableKind(kind)
    contract = CONTRACTS[kind]
    errors: list[str] = []

    _check_missing_columns(df, contract, errors)
    if errors:
        raise DataContractError(f"{kind.value} contract failed: " + "; ".join(errors))

    out = _normalize_unknowns(df)

    if contract.require_unique_id:
        id_col = SCHEMA.item_id if kind == TableKind.ITEMS else SCHEMA.term_id if kind == TableKind.TERMS else SCHEMA.id
        _check_unique_id(out, id_col, errors)
    if contract.require_unique_pair:
        _check_unique_pair(out, errors)
    if contract.require_label:
        _check_label(out, errors)
    if positive_only:
        _check_positive_only_training_pairs(out, errors)
    if contract.require_no_empty_query:
        _check_empty_text(out, SCHEMA.query, errors)
    if contract.require_no_empty_title:
        _check_empty_text(out, SCHEMA.title, errors)

    if errors:
        raise DataContractError(f"{kind.value} contract failed: " + "; ".join(errors))
    return out
