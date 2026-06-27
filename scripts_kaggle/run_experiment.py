#!/usr/bin/env python
"""OOF-first experiment runner with registry and threshold reports."""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.data.io import read_table
from src_kaggle.experiments.experiment_registry import append_experiment
from src_kaggle.experiments.oof_manager import save_experiment_oof, snapshot_config, standardize_oof
from src_kaggle.utils.config import load_kaggle_config
from src_kaggle.validation.private_lb_simulator import run_validation
from src_kaggle.validation.threshold_optimizer import optimize_oof_thresholds, write_threshold_report


def parse_args():
    p = argparse.ArgumentParser(description="Run OOF-first Kaggle experiment")
    p.add_argument("--config", default="configs/kaggle/war_mode.yaml")
    p.add_argument("--name", default=None)
    p.add_argument("--model-type", choices=["tabular", "cross_encoder"], default=None)
    p.add_argument("--train", default=None)
    p.add_argument("--items", default=None)
    p.add_argument("--public-lb-score", default="")
    p.add_argument("--note", default="")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_kaggle_config(args.config)
    exp_name = args.name or f"{args.model_type or cfg.get('validation_framework',{}).get('model_type','tabular')}_{uuid.uuid4().hex[:8]}"
    exp_id = uuid.uuid4().hex[:12]
    exp_dir = Path("artifacts/experiments") / exp_name
    report_dir = Path("reports/experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)

    val_cfg = dict(cfg.get("validation_framework", {}))
    if args.model_type:
        val_cfg["model_type"] = args.model_type
    if "model" not in val_cfg or not val_cfg.get("model"):
        val_cfg["model"] = cfg.get("modeling", {}).get(val_cfg.get("model_type", "tabular"), {})
    if "features" not in val_cfg or not val_cfg.get("features"):
        val_cfg["features"] = cfg.get("feature_engineering", {})

    df = read_table(args.train or cfg["paths"].get("train_model_input") or cfg["paths"].get("train_merged"))
    items = read_table(args.items or cfg["paths"]["items"])
    summary, last = run_validation(df, items, val_cfg, str(report_dir / "validation"))
    oof = last["oof"]
    opt = optimize_oof_thresholds(oof, segment_min_rows=int(val_cfg.get("segment_min_rows", 30)))
    threshold_summary = write_threshold_report(opt, report_dir / "threshold")
    oof_std = standardize_oof(oof, exp_name, val_cfg.get("model_type", "tabular"), opt.best_threshold)
    oof_path = save_experiment_oof(oof_std, exp_dir)
    snapshot_config({"base_config": cfg, "validation_config": val_cfg, "threshold_summary": threshold_summary}, exp_dir)

    record = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "model_type": val_cfg.get("model_type", "tabular"),
        "backbone": val_cfg.get("model", {}).get("model_name", ""),
        "booster_type": val_cfg.get("model", {}).get("model_type", ""),
        "data_version": cfg.get("paths", {}).get("train_model_input", ""),
        "negative_mining_version": "config:negative_mining",
        "retrieval_version": "config:retrieval",
        "feature_version": "config:feature_engineering",
        "validation_version": val_cfg.get("splitter", "term_group"),
        "seed": ",".join(map(str, val_cfg.get("seeds", [val_cfg.get("seed", 42)]))),
        "fold_count": val_cfg.get("n_folds", 5),
        "oof_macro_f1": opt.best_metrics["macro_f1"],
        "class0_f1": opt.best_metrics["class0_f1"],
        "class1_f1": opt.best_metrics["class1_f1"],
        "best_threshold": opt.best_threshold,
        "threshold_fragile": opt.sensitivity.get("is_fragile", False),
        "oof_path": oof_path,
        "report_dir": str(report_dir),
        "submission_note": args.note,
        "public_lb_score": args.public_lb_score,
    }
    append_experiment(record)
    print(f"[OK] experiment={exp_name} macro_f1={record['oof_macro_f1']:.6f} threshold={opt.best_threshold:.3f}")
    print(f"[OK] oof={oof_path}")
    print(f"[OK] reports={report_dir}")


if __name__ == "__main__":
    main()
