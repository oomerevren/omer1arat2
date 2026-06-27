"""Strict config loading for Kaggle War Mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src_kaggle.data.schema import CANONICAL_FIELD_MAP, SCHEMA


class KaggleConfigError(ValueError):
    """Raised when a config is unsafe for Kaggle War Mode."""


def load_kaggle_config(path: str | Path = "configs/kaggle/war_mode.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    validate_kaggle_config(cfg, source=str(path))
    return cfg


def validate_kaggle_config(cfg: dict[str, Any], *, source: str = "<dict>") -> None:
    mode = cfg.get("mode") or cfg.get("experiment", {}).get("mode")
    problem_type = cfg.get("problem", {}).get("type")
    metric = cfg.get("validation", {}).get("metric")
    num_labels = cfg.get("model", {}).get("num_labels")

    errors: list[str] = []
    if mode != "kaggle_war_mode":
        errors.append(f"mode must be 'kaggle_war_mode', got {mode!r}")
    if problem_type != "binary_pair_classification":
        errors.append(f"problem.type must be 'binary_pair_classification', got {problem_type!r}")
    if num_labels != 2:
        errors.append(f"model.num_labels must be 2 for Kaggle, got {num_labels!r}")
    if metric != "macro_f1":
        errors.append(f"validation.metric must be 'macro_f1', got {metric!r}")

    columns = cfg.get("data", {}).get("columns", {})
    for key, expected in CANONICAL_FIELD_MAP.items():
        if columns.get(key) != expected:
            errors.append(f"data.columns.{key} must be {expected!r}, got {columns.get(key)!r}")

    canonical_fields = cfg.get("canonical_fields", {})
    for key, expected in CANONICAL_FIELD_MAP.items():
        if canonical_fields.get(key) != expected:
            errors.append(f"canonical_fields.{key} must be {expected!r}, got {canonical_fields.get(key)!r}")

    paths = cfg.get("paths", {})
    for key in ("items", "terms", "training_pairs", "submission_pairs", "sample_submission", "train_merged", "test_merged"):
        if key not in paths:
            errors.append(f"paths.{key} is required")

    contract = cfg.get("data_contract", {})
    expected_contracts = {
        "items_csv": [SCHEMA.item_id, SCHEMA.title, SCHEMA.category, SCHEMA.brand, SCHEMA.gender, SCHEMA.age_group, SCHEMA.attributes],
        "terms_csv": [SCHEMA.term_id, SCHEMA.query],
        "training_pairs_csv": [SCHEMA.id, SCHEMA.term_id, SCHEMA.item_id, SCHEMA.label],
        "submission_pairs_csv": [SCHEMA.id, SCHEMA.term_id, SCHEMA.item_id],
        "sample_submission_csv": [SCHEMA.id, SCHEMA.prediction],
    }
    for name, required in expected_contracts.items():
        got = contract.get(name, {}).get("required_columns")
        if got != required:
            errors.append(f"data_contract.{name}.required_columns must be {required!r}, got {got!r}")

    forbidden = cfg.get("forbidden_in_kaggle", {})
    enabled_forbidden = [name for name, enabled in forbidden.items() if bool(enabled)]
    if enabled_forbidden:
        errors.append(f"forbidden Kaggle features enabled: {enabled_forbidden}")

    if errors:
        joined = "\n  - ".join(errors)
        raise KaggleConfigError(f"Unsafe Kaggle config: {source}\n  - {joined}")
