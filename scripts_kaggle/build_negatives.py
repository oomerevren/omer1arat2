#!/usr/bin/env python
"""Build multi-layer hard negatives and augmented training set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.data.io import read_table
from src_kaggle.data.negative_mining import NegativeMiningConfig, run_negative_mining_and_write
from src_kaggle.retrieval.semantic_confuser_analysis import write_semantic_confuser_report
from src_kaggle.data.schema import SCHEMA
from src_kaggle.utils.config import load_kaggle_config


def parse_args():
    p = argparse.ArgumentParser(description="Build Kaggle hard negatives")
    p.add_argument("--config", default="configs/kaggle/war_mode.yaml")
    p.add_argument("--train-merged", default=None)
    p.add_argument("--items", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--uncertain-output", default=None)
    p.add_argument("--report-json", default=None)
    p.add_argument("--report-md", default=None)
    p.add_argument("--mode", choices=["global", "fold_aware"], default=None)
    p.add_argument("--active-fold", type=int, default=None)
    p.add_argument("--use-dense", default=None, help="true/false: activate dense pool")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_kaggle_config(args.config)
    nm_cfg_dict = dict(cfg.get("negative_mining", {}))
    if args.mode:
        nm_cfg_dict["mode"] = args.mode
    if args.active_fold is not None:
        nm_cfg_dict["active_fold"] = args.active_fold
    if args.use_dense is not None:
        nm_cfg_dict["use_dense_pool"] = str(args.use_dense).lower() in {"1", "true", "yes", "y"}
        if nm_cfg_dict["use_dense_pool"] and int(nm_cfg_dict.get("dense_negatives_per_positive", 0)) <= 0:
            nm_cfg_dict["dense_negatives_per_positive"] = 2
    nm_cfg_dict["retrieval_cfg"] = cfg.get("retrieval", {})
    nm_cfg = NegativeMiningConfig.from_dict(nm_cfg_dict)

    paths = cfg["paths"]
    reports = cfg.get("reports", {})
    positives = read_table(args.train_merged or paths.get("train_merged"))
    items = read_table(args.items or paths.get("items"))

    augmented, uncertain, report = run_negative_mining_and_write(
        positives=positives,
        items=items,
        cfg=nm_cfg,
        output_path=args.output or paths.get("train_model_input", "data/processed/train_with_negatives.parquet"),
        uncertain_path=args.uncertain_output or paths.get("uncertain_negatives", "data/processed/uncertain_negative_candidates.parquet"),
        report_json_path=args.report_json or reports.get("negative_mining_json", "reports/negative_mining/negative_mining_report.json"),
        report_md_path=args.report_md or reports.get("negative_mining_md", "reports/negative_mining/negative_mining_report.md"),
    )
    confuser_path = reports.get("semantic_confuser_examples", "reports/retrieval/semantic_confuser_examples.md")
    negatives = augmented[augmented.get(SCHEMA.label, 1).eq(0)] if SCHEMA.label in augmented.columns else augmented.iloc[0:0]
    write_semantic_confuser_report(negatives, uncertain, confuser_path)
    print(f"[OK] augmented training rows={len(augmented)}")
    print(f"[OK] uncertain candidates={len(uncertain)}")
    print(f"[OK] negatives={report['total_negatives']}")
    print(f"[OK] semantic confuser report={confuser_path}")


if __name__ == "__main__":
    main()
