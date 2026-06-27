from __future__ import annotations
from pathlib import Path
import pandas as pd

WATCH_SEGMENTS = ["is_short_query","is_long_query","is_brand_heavy","is_attribute_heavy","is_category_heavy","has_gender_token","has_age_token","negative_type","source_pool","gender","age_group"]


def collect_segment_risk(df: pd.DataFrame, min_macro: float = 0.45) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        report_dir = str(r.get("report_dir", ""))
        candidates = [Path(report_dir)/"validation"/"segment_scores.csv", Path(report_dir)/"segment_scores.csv"] if report_dir else []
        for p in candidates:
            if not p.exists():
                continue
            seg=pd.read_csv(p)
            for _, s in seg.iterrows():
                macro=float(s.get("macro_f1", 0) or 0)
                c0=float(s.get("class0_f1", 0) or 0)
                collapse = macro < min_macro or c0 < min_macro
                if str(s.get("segment", "")) in WATCH_SEGMENTS or collapse:
                    rows.append({
                        "experiment_id": r.get("experiment_id"), "experiment_name": r.get("experiment_name"),
                        "model_family": r.get("model_family"), "segment": s.get("segment"), "value": s.get("value"),
                        "n": s.get("n"), "macro_f1": macro, "class0_f1": c0, "class1_f1": s.get("class1_f1"),
                        "segment_collapse_flag": collapse,
                    })
    return pd.DataFrame(rows)


def write_segment_risk(table: pd.DataFrame, out_dir: str|Path):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if table.empty:
        table=pd.DataFrame(columns=["experiment_id","experiment_name","model_family","segment","value","n","macro_f1","class0_f1","class1_f1","segment_collapse_flag"])
    table.to_csv(out/"segment_risk_report.csv", index=False)
