"""Pair-centric merged dataset builder for Kaggle War Mode.

The competition task is pair relevance, not item classification. This module
therefore builds the canonical model input by joining pair tables with terms and
items without silently dropping rows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src_kaggle.data.attribute_parser import add_attribute_features
from src_kaggle.data.contracts import DataContractError, TableKind, validate_dataframe
from src_kaggle.data.io import read_contract_table, write_table
from src_kaggle.data.schema import SCHEMA, TEST_MERGED_COLUMNS, TRAIN_MERGED_COLUMNS
from src_kaggle.features.query_intent import add_query_intent_features, build_query_intent_resources

SplitName = Literal["train", "test"]


@dataclass(frozen=True)
class PairBuildReport:
    split: str
    input_pairs: int
    output_rows: int
    successful_join_rows: int
    missing_term_id_rows: int
    missing_item_id_rows: int
    null_query_rows: int
    null_title_rows: int
    duplicate_id_rows: int
    duplicate_pair_rows: int
    unique_terms_in_pairs: int
    unique_items_in_pairs: int
    unique_missing_term_ids: int
    unique_missing_item_ids: int
    output_path: str | None = None

    @property
    def has_join_issues(self) -> bool:
        return any(
            value > 0
            for value in (
                self.missing_term_id_rows,
                self.missing_item_id_rows,
                self.null_query_rows,
                self.null_title_rows,
                self.duplicate_id_rows,
                self.duplicate_pair_rows,
            )
        )


def _empty_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def build_full_item_text(items: pd.DataFrame) -> pd.DataFrame:
    """Create a retrieval/model friendly item text field from canonical item columns."""
    out = items.copy()
    if SCHEMA.normalized_attribute_text not in out.columns:
        out = add_attribute_features(out)
    fields = [
        SCHEMA.title,
        SCHEMA.category,
        SCHEMA.brand,
        SCHEMA.gender,
        SCHEMA.age_group,
        SCHEMA.normalized_attribute_text,
    ]
    out[SCHEMA.full_item_text] = out[fields].fillna("").astype(str).agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def _quality_report(split: SplitName, pairs: pd.DataFrame, merged: pd.DataFrame, output_path: str | None = None) -> PairBuildReport:
    missing_term = merged[SCHEMA.query].isna() if SCHEMA.query in merged.columns else pd.Series([True] * len(merged))
    missing_item = merged[SCHEMA.title].isna() if SCHEMA.title in merged.columns else pd.Series([True] * len(merged))
    duplicate_id_rows = int(pairs.duplicated(SCHEMA.id, keep=False).sum()) if SCHEMA.id in pairs.columns else 0
    duplicate_pair_rows = int(pairs.duplicated([SCHEMA.term_id, SCHEMA.item_id], keep=False).sum())

    missing_term_ids = merged.loc[missing_term, SCHEMA.term_id].dropna().unique().tolist() if SCHEMA.term_id in merged.columns else []
    missing_item_ids = merged.loc[missing_item, SCHEMA.item_id].dropna().unique().tolist() if SCHEMA.item_id in merged.columns else []

    return PairBuildReport(
        split=split,
        input_pairs=int(len(pairs)),
        output_rows=int(len(merged)),
        successful_join_rows=int((~missing_term & ~missing_item).sum()),
        missing_term_id_rows=int(missing_term.sum()),
        missing_item_id_rows=int(missing_item.sum()),
        null_query_rows=int(_empty_mask(merged[SCHEMA.query]).sum()) if SCHEMA.query in merged.columns else int(len(merged)),
        null_title_rows=int(_empty_mask(merged[SCHEMA.title]).sum()) if SCHEMA.title in merged.columns else int(len(merged)),
        duplicate_id_rows=duplicate_id_rows,
        duplicate_pair_rows=duplicate_pair_rows,
        unique_terms_in_pairs=int(pairs[SCHEMA.term_id].nunique()),
        unique_items_in_pairs=int(pairs[SCHEMA.item_id].nunique()),
        unique_missing_term_ids=int(len(missing_term_ids)),
        unique_missing_item_ids=int(len(missing_item_ids)),
        output_path=output_path,
    )


def _merge_pairs(pairs: pd.DataFrame, terms: pd.DataFrame, items: pd.DataFrame, split: SplitName) -> tuple[pd.DataFrame, PairBuildReport]:
    terms = validate_dataframe(terms, TableKind.TERMS)
    items = build_full_item_text(validate_dataframe(items, TableKind.ITEMS))

    if split == "train":
        pairs = validate_dataframe(pairs, TableKind.TRAINING_PAIRS, positive_only=True)
        output_columns = TRAIN_MERGED_COLUMNS
    else:
        pairs = validate_dataframe(pairs, TableKind.SUBMISSION_PAIRS)
        output_columns = TEST_MERGED_COLUMNS

    term_cols = [SCHEMA.term_id, SCHEMA.query]
    item_cols = [
        SCHEMA.item_id,
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
    ]

    merged = pairs.merge(terms[term_cols], on=SCHEMA.term_id, how="left", validate="many_to_one")
    merged = merged.merge(items[item_cols], on=SCHEMA.item_id, how="left", validate="many_to_one")
    resources = build_query_intent_resources(items)
    merged = add_query_intent_features(merged, resources)
    # Keep canonical columns first, then append query-intent columns for downstream
    # feature engineering and segment-level diagnostics.
    intent_cols = [col for col in merged.columns if col not in output_columns]
    merged = merged.loc[:, list(output_columns) + intent_cols]
    report = _quality_report(split, pairs, merged)
    return merged, report


def build_train_pairs(training_pairs: pd.DataFrame, terms: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, PairBuildReport]:
    return _merge_pairs(training_pairs, terms, items, "train")


def build_submission_pairs(submission_pairs: pd.DataFrame, terms: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, PairBuildReport]:
    return _merge_pairs(submission_pairs, terms, items, "test")


def write_report(report: PairBuildReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")


def print_report(report: PairBuildReport) -> None:
    print(f"[{report.split}] input_pairs={report.input_pairs} output_rows={report.output_rows}")
    print(f"[{report.split}] successful_join_rows={report.successful_join_rows}")
    print(f"[{report.split}] missing_term_id_rows={report.missing_term_id_rows} missing_item_id_rows={report.missing_item_id_rows}")
    print(f"[{report.split}] null_query_rows={report.null_query_rows} null_title_rows={report.null_title_rows}")
    print(f"[{report.split}] duplicate_id_rows={report.duplicate_id_rows} duplicate_pair_rows={report.duplicate_pair_rows}")


def assert_report_clean(report: PairBuildReport) -> None:
    if report.has_join_issues:
        raise DataContractError(
            f"{report.split} pair build has data quality issues. "
            "See reports/data_quality/*_pair_build_report.json for details."
        )


def build_pair_datasets_from_paths(
    *,
    training_pairs_path: str | Path,
    submission_pairs_path: str | Path,
    terms_path: str | Path,
    items_path: str | Path,
    train_output_path: str | Path,
    test_output_path: str | Path,
    train_report_path: str | Path,
    test_report_path: str | Path,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, PairBuildReport, PairBuildReport]:
    """Read official files, build merged pair datasets, write outputs and reports."""
    training_pairs = read_contract_table(training_pairs_path, TableKind.TRAINING_PAIRS, positive_only=True)
    submission_pairs = read_contract_table(submission_pairs_path, TableKind.SUBMISSION_PAIRS)
    terms = read_contract_table(terms_path, TableKind.TERMS)
    items = read_contract_table(items_path, TableKind.ITEMS)

    train_df, train_report = build_train_pairs(training_pairs, terms, items)
    test_df, test_report = build_submission_pairs(submission_pairs, terms, items)
    train_report = PairBuildReport(**{**asdict(train_report), "output_path": str(train_output_path)})
    test_report = PairBuildReport(**{**asdict(test_report), "output_path": str(test_output_path)})

    write_table(train_df, train_output_path)
    write_table(test_df, test_output_path)
    write_report(train_report, train_report_path)
    write_report(test_report, test_report_path)
    print_report(train_report)
    print_report(test_report)

    if strict:
        assert_report_clean(train_report)
        assert_report_clean(test_report)

    return train_df, test_df, train_report, test_report
