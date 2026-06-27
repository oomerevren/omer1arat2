"""OOF error analysis pipeline with taxonomy and action suggestions."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src_kaggle.analysis.error_taxonomy import assign_error_tags, action_suggestions
from src_kaggle.data.schema import SCHEMA


def analyze_errors(oof: pd.DataFrame, threshold: float | None = None, top_n: int = 200) -> dict:
    df = oof.copy()
    threshold = float(threshold if threshold is not None else 0.5)
    if "pred_best_threshold" not in df.columns:
        df["pred_best_threshold"] = (df["proba"] >= threshold).astype(int)
    df["error_type"] = "correct"
    df.loc[(df[SCHEMA.label] == 0) & (df["pred_best_threshold"] == 1), "error_type"] = "false_positive"
    df.loc[(df[SCHEMA.label] == 1) & (df["pred_best_threshold"] == 0), "error_type"] = "false_negative"
    err = df[df["error_type"] != "correct"].copy()
    if not err.empty:
        err["error_tags"] = err.apply(lambda r: "|".join(assign_error_tags(r, threshold)), axis=1)
        err["confidence_error_score"] = err.apply(lambda r: r["proba"] if r["error_type"]=="false_positive" else 1-r["proba"], axis=1)
    else:
        err["error_tags"] = []
        err["confidence_error_score"] = []
    tag_counts = err["error_tags"].str.get_dummies(sep="|").sum().sort_values(ascending=False).to_dict() if not err.empty else {}
    segment_cols = [c for c in ["is_short_query","is_brand_heavy","is_attribute_heavy","is_category_heavy","has_gender_token","has_age_token","negative_type","source_pool",SCHEMA.category,SCHEMA.gender,SCHEMA.age_group] if c in err.columns]
    segment_counts = {c: err[c].value_counts(dropna=False).head(20).to_dict() for c in segment_cols}
    fp = err[err["error_type"]=="false_positive"].sort_values("confidence_error_score", ascending=False).head(top_n)
    fn = err[err["error_type"]=="false_negative"].sort_values("confidence_error_score", ascending=False).head(top_n)
    return {"annotated": err, "top_fp": fp, "top_fn": fn, "summary": {"total_rows": int(len(df)), "total_errors": int(len(err)), "false_positives": int(len(fp)), "false_negatives": int(len(fn)), "error_tag_counts": tag_counts, "segment_error_counts": segment_counts, "action_suggestions": action_suggestions(tag_counts)}}


def write_error_reports(result: dict, out_dir: str | Path = "reports/errors") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    result["top_fp"].to_csv(out/"top_false_positives.csv", index=False)
    result["top_fn"].to_csv(out/"top_false_negatives.csv", index=False)
    result["annotated"].to_csv(out/"annotated_errors.csv", index=False)
    (out/"error_summary.json").write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines=["# Error Summary", "", f"Total rows: {result['summary']['total_rows']}", f"Total errors: {result['summary']['total_errors']}", "", "## Error tag counts"]
    for k,v in result["summary"]["error_tag_counts"].items(): lines.append(f"- {k}: {v}")
    lines += ["", "## Action suggestions"]
    for s in result["summary"]["action_suggestions"]: lines.append(f"- {s}")
    (out/"error_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"top_fp": str(out/"top_false_positives.csv"), "top_fn": str(out/"top_false_negatives.csv"), "summary": str(out/"error_summary.json")}
