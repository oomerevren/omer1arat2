"""Executable OOF-first ablation runner.

The runner is conservative: official Kaggle files are required for real metrics.
When data/dependency constraints prevent a run, it writes a structured not-run row
instead of fabricating scores.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src_kaggle.ablation.ablation_specs import AblationSpec, all_ablation_specs
from src_kaggle.ablation.component_toggle import apply_toggles
from src_kaggle.ablation.ablation_reporting import finalize_master, write_ablation_reports
from src_kaggle.data.io import read_table
from src_kaggle.data.schema import SCHEMA
from src_kaggle.validation.private_lb_simulator import run_validation
from src_kaggle.validation.threshold_optimizer import optimize_oof_thresholds, write_threshold_report
from src_kaggle.validation.threshold_tuning import metrics_at_threshold


def _data_available(cfg: dict[str, Any]) -> tuple[bool, str]:
    paths = cfg.get("paths", {})
    required = [paths.get("items"), paths.get("train_model_input") or paths.get("train_merged")]
    missing = [str(p) for p in required if not p or not Path(p).exists()]
    if missing:
        return False, "missing_data: " + ",".join(missing)
    return True, ""


def _seed_std(summary: dict[str, Any]) -> float:
    vals = [float(x.get("macro_f1", 0)) for x in summary.get("seed_results", [])]
    return float(np.std(vals)) if vals else 0.0


def _threshold_metrics(oof: pd.DataFrame, strategy: str, opt) -> tuple[float, dict[str, float], str]:
    if strategy == "fixed_05":
        t = 0.5
    elif strategy == "class0_protective":
        t = min(0.95, float(opt.best_threshold) + 0.05)
    elif strategy == "class1_protective":
        t = max(0.05, float(opt.best_threshold) - 0.05)
    elif strategy == "stable_midpoint":
        curve = opt.curve
        best = float(curve["macro_f1"].max()) if not curve.empty else 0.0
        near = curve[curve["macro_f1"] >= best - 0.002]
        t = float((near["threshold"].min() + near["threshold"].max()) / 2) if not near.empty else float(opt.best_threshold)
    else:
        t = float(opt.best_threshold)
    m = metrics_at_threshold(oof[SCHEMA.label], oof["proba"], t)
    note = f"threshold_strategy={strategy}; analysis_only" if strategy == "segment_analysis_only" else f"threshold_strategy={strategy}"
    return float(t), m, note


class AblationRunner:
    def __init__(self, base_cfg: dict[str, Any], out_dir: str | Path = "reports/ablation", max_runs: int | None = None, smoke: bool = False, allow_transformer: bool = False, allow_real_dense: bool = False):
        self.base_cfg = base_cfg
        self.out_dir = Path(out_dir)
        self.max_runs = max_runs
        self.smoke = smoke
        self.allow_transformer = allow_transformer
        self.allow_real_dense = allow_real_dense

    def _not_run_row(self, spec: AblationSpec, reason: str) -> dict[str, Any]:
        val = self.base_cfg.get("validation_framework", {})
        return {
            "ablation_id": spec.ablation_id,
            "experiment_name": spec.experiment_name,
            "category": spec.category,
            "changed_component": spec.changed_component,
            "variant_description": spec.variant_description,
            "splitter": val.get("splitter", "term_group"),
            "seed_set": ",".join(map(str, val.get("seeds", [val.get("seed", 42)]))),
            "OOF macro-F1": math.nan,
            "class0 F1": math.nan,
            "class1 F1": math.nan,
            "best_threshold": math.nan,
            "threshold_fragility": "",
            "seed_std": math.nan,
            "public_lb_score": "",
            "public_oof_delta": "",
            "note": reason,
            "risk_flag": "needs_fold_safe_recheck",
            "status": "not_run",
            "report_dir": "",
            "oof_path": "",
        }

    def run_spec(self, spec: AblationSpec) -> dict[str, Any]:
        if spec.requires_transformer and not self.allow_transformer:
            return self._not_run_row(spec, "requires real transformer/GPU OOF; skipped unless --allow-transformer")
        if spec.requires_real_dense and not self.allow_real_dense:
            return self._not_run_row(spec, "requires real_dense artefacts; skipped unless --allow-real-dense")
        cfg = apply_toggles(self.base_cfg, spec.toggles)
        ok, reason = _data_available(cfg)
        if not ok:
            return self._not_run_row(spec, reason)
        if spec.requires_negative_rebuild:
            # Negative rebuild is intentionally not hidden inside ablation.  It must
            # be run explicitly so data versions are auditable.
            return self._not_run_row(spec, "requires negative dataset rebuild for this variant; run build_negatives with exported config snapshot first")

        paths = cfg["paths"]
        train_path = paths.get("train_model_input") or paths.get("train_merged")
        df = read_table(train_path)
        items = read_table(paths["items"])
        if self.smoke:
            df = df.head(min(len(df), 500)).copy()
            # Need both classes for OOF; smoke mode is only for code health.
            if SCHEMA.label in df and df[SCHEMA.label].nunique() < 2:
                return self._not_run_row(spec, "smoke sample lacks both classes")
        val_cfg = dict(cfg.get("validation_framework", {}))
        val_cfg["features"] = cfg.get("feature_engineering", {})
        if spec.toggles.get("model_type"):
            val_cfg["model_type"] = spec.toggles["model_type"]
        if val_cfg.get("model_type") == "cross_encoder":
            val_cfg["model"] = cfg.get("modeling", {}).get("cross_encoder", {})
        else:
            val_cfg["model"] = cfg.get("modeling", {}).get("tabular", val_cfg.get("model", {}))
        exp_dir = self.out_dir / "runs" / spec.experiment_name
        summary, last = run_validation(df, items, val_cfg, exp_dir / "validation")
        oof = last["oof"]
        opt = optimize_oof_thresholds(oof, segment_min_rows=int(val_cfg.get("segment_min_rows", 30)))
        threshold_summary = write_threshold_report(opt, exp_dir / "threshold")
        strategy = cfg.get("threshold_optimization", {}).get("strategy", "oof_best")
        t, metrics, extra_note = _threshold_metrics(oof, strategy, opt)
        (exp_dir / "ablation_config_snapshot.json").write_text(json.dumps({"spec": spec.to_dict(), "config": cfg, "threshold_summary": threshold_summary}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        oof_path = exp_dir / "oof_predictions.csv"
        oof.to_csv(oof_path, index=False)
        return {
            "ablation_id": spec.ablation_id,
            "experiment_name": spec.experiment_name,
            "category": spec.category,
            "changed_component": spec.changed_component,
            "variant_description": spec.variant_description,
            "splitter": val_cfg.get("splitter", "term_group"),
            "seed_set": ",".join(map(str, val_cfg.get("seeds", [val_cfg.get("seed", 42)]))),
            "OOF macro-F1": metrics["macro_f1"],
            "class0 F1": metrics["class0_f1"],
            "class1 F1": metrics["class1_f1"],
            "best_threshold": t,
            "threshold_fragility": opt.sensitivity.get("is_fragile", False),
            "seed_std": _seed_std(summary),
            "public_lb_score": "",
            "public_oof_delta": "",
            "note": "; ".join(x for x in [spec.risk_note, extra_note] if x),
            "risk_flag": "",
            "status": "completed",
            "report_dir": str(exp_dir),
            "oof_path": str(oof_path),
        }

    def run(self, specs: list[AblationSpec] | None = None, categories: set[str] | None = None) -> pd.DataFrame:
        specs = specs or all_ablation_specs()
        if categories:
            specs = [s for s in specs if s.category in categories]
        if self.max_runs is not None:
            specs = specs[: self.max_runs]
        rows = [self.run_spec(s) for s in specs]
        master = finalize_master(rows)
        write_ablation_reports(master, self.out_dir)
        (self.out_dir / "ablation_specs.json").write_text(json.dumps([s.to_dict() for s in specs], indent=2, ensure_ascii=False), encoding="utf-8")
        return master
