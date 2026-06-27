"""Strict Kaggle submission validation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd

REQUIRED_COLUMNS = ["id", "prediction"]

@dataclass
class SubmissionValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    n_rows: int
    positive_rate: float
    output_path: str | None = None

    def to_dict(self): return asdict(self)


def validate_submission(
    submission: pd.DataFrame,
    reference_pairs: pd.DataFrame,
    *,
    previous_submission: pd.DataFrame | None = None,
    min_positive_rate: float = 0.001,
    max_positive_rate: float = 0.999,
    expected_order: bool = True,
) -> SubmissionValidationResult:
    errors=[]; warnings=[]
    if list(submission.columns) != REQUIRED_COLUMNS:
        errors.append(f"columns must be exactly {REQUIRED_COLUMNS}, got {list(submission.columns)}")
    if "id" not in submission.columns or "prediction" not in submission.columns:
        return SubmissionValidationResult(False, errors or ["missing id/prediction"], warnings, len(submission), 0.0)
    if len(submission) != len(reference_pairs): errors.append(f"row count mismatch submission={len(submission)} reference={len(reference_pairs)}")
    if submission["id"].duplicated().any(): errors.append("duplicate id in submission")
    if reference_pairs["id"].duplicated().any(): errors.append("duplicate id in reference_pairs")
    sub_ids=set(submission["id"].tolist()); ref_ids=set(reference_pairs["id"].tolist())
    if sub_ids != ref_ids:
        errors.append(f"id set mismatch missing={len(ref_ids-sub_ids)} extra={len(sub_ids-ref_ids)}")
    if expected_order and len(submission)==len(reference_pairs) and not submission["id"].reset_index(drop=True).equals(reference_pairs["id"].reset_index(drop=True)):
        errors.append("id order mismatch vs submission_pairs/sample reference")
    if submission["prediction"].isna().any(): errors.append("null prediction exists")
    invalid=sorted(set(submission["prediction"].dropna().unique()) - {0,1,"0","1"})
    if invalid: errors.append(f"prediction must be binary 0/1, invalid={invalid[:10]}")
    pred=pd.to_numeric(submission["prediction"], errors="coerce")
    positive_rate=float(pred.mean()) if len(pred) else 0.0
    if positive_rate < min_positive_rate or positive_rate > max_positive_rate:
        warnings.append(f"positive_rate={positive_rate:.6f} outside [{min_positive_rate},{max_positive_rate}]")
    if previous_submission is not None and {"id","prediction"}.issubset(previous_submission.columns):
        prev=previous_submission.set_index("id").reindex(submission["id"])["prediction"]
        diff=float((pd.to_numeric(prev,errors="coerce").values != pred.values).mean()) if len(pred) else 0.0
        if diff > 0.25: warnings.append(f"dramatic change vs previous submission diff_rate={diff:.3f}")
    return SubmissionValidationResult(len(errors)==0, errors, warnings, len(submission), positive_rate)


def write_validation_report(result: SubmissionValidationResult, path: str | Path) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
