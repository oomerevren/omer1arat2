"""Safe submission builder and registry."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from src_kaggle.submission.submission_validator import validate_submission, write_validation_report


def build_submission_from_proba(reference_pairs: pd.DataFrame, proba, threshold: float) -> pd.DataFrame:
    out=pd.DataFrame({"id": reference_pairs["id"].values, "prediction": (pd.Series(proba).values >= threshold).astype(int)})
    return out[["id","prediction"]]


def build_submission_from_predictions(reference_pairs: pd.DataFrame, predictions) -> pd.DataFrame:
    out=pd.DataFrame({"id": reference_pairs["id"].values, "prediction": pd.Series(predictions).astype(int).values})
    return out[["id","prediction"]]


def save_safe_submission(
    submission: pd.DataFrame,
    reference_pairs: pd.DataFrame,
    output_path: str | Path,
    *,
    report_path: str | Path,
    previous_submission: pd.DataFrame | None = None,
    fail_on_warning: bool = False,
) -> dict:
    result=validate_submission(submission, reference_pairs, previous_submission=previous_submission)
    result.output_path=str(output_path)
    write_validation_report(result, report_path)
    if not result.is_valid:
        unsafe=Path(str(output_path)+".unsafe.csv"); unsafe.parent.mkdir(parents=True, exist_ok=True); submission.to_csv(unsafe,index=False)
        raise ValueError(f"Unsafe submission, not saved as final. Errors={result.errors}. Unsafe copy={unsafe}")
    if fail_on_warning and result.warnings:
        raise ValueError(f"Submission warnings with fail_on_warning=True: {result.warnings}")
    output_path=Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path,index=False)
    return result.to_dict()


def append_submission_registry(record: dict, registry_path: str | Path = "reports/submissions/submission_registry.csv") -> pd.DataFrame:
    path=Path(registry_path); path.parent.mkdir(parents=True, exist_ok=True)
    row={
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": record.get("experiment_id", ""),
        "models": record.get("models", ""),
        "threshold": record.get("threshold", ""),
        "validation_score": record.get("validation_score", ""),
        "class0_f1": record.get("class0_f1", ""),
        "class1_f1": record.get("class1_f1", ""),
        "positive_rate": record.get("positive_rate", ""),
        "file_path": record.get("file_path", ""),
        "validation_report": record.get("validation_report", ""),
        "public_lb_score": record.get("public_lb_score", ""),
        "note": record.get("note", ""),
    }
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    df=pd.concat([old,pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path,index=False)
    return df
