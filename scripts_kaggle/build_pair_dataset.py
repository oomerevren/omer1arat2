#!/usr/bin/env python
"""Build pair-centric merged datasets for Kaggle War Mode.

Expected official files:
  items.csv: item_id,title,category,brand,gender,age_group,attributes
  terms.csv: term_id,query
  training_pairs.csv: id,term_id,item_id,label (label is always 1)
  submission_pairs.csv: id,term_id,item_id

Outputs are written to data/processed/ by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.data.pair_builder import build_pair_datasets_from_paths
from src_kaggle.utils.config import load_kaggle_config


def parse_args():
    p = argparse.ArgumentParser(description="Build pair-centric merged Kaggle datasets")
    p.add_argument("--config", default="configs/kaggle/war_mode.yaml")
    p.add_argument("--train", default=None, help="official training_pairs.csv")
    p.add_argument("--test", default=None, help="official submission_pairs.csv")
    p.add_argument("--terms", default=None, help="official terms.csv")
    p.add_argument("--items", default=None, help="official items.csv")
    p.add_argument("--out-train", default=None, help="merged train output (.csv or .parquet)")
    p.add_argument("--out-test", default=None, help="merged test output (.csv or .parquet)")
    p.add_argument("--train-report", default=None)
    p.add_argument("--test-report", default=None)
    p.add_argument("--no-strict", action="store_true", help="write outputs/reports even if join quality issues exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_kaggle_config(args.config)
    paths = cfg["paths"]
    report_paths = cfg.get("reports", {})

    build_pair_datasets_from_paths(
        training_pairs_path=args.train or paths["training_pairs"],
        submission_pairs_path=args.test or paths["submission_pairs"],
        terms_path=args.terms or paths["terms"],
        items_path=args.items or paths["items"],
        train_output_path=args.out_train or paths["train_merged"],
        test_output_path=args.out_test or paths["test_merged"],
        train_report_path=args.train_report or report_paths.get("train_pair_build", "reports/data_quality/train_pair_build_report.json"),
        test_report_path=args.test_report or report_paths.get("test_pair_build", "reports/data_quality/test_pair_build_report.json"),
        strict=not args.no_strict,
    )
    print("[OK] pair-centric merged datasets built")


if __name__ == "__main__":
    main()
