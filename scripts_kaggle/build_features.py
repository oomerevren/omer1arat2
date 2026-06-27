#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.data.io import read_table, write_table
from src_kaggle.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig, write_feature_reports
from src_kaggle.utils.config import load_kaggle_config


def parse_args():
    p=argparse.ArgumentParser(description="Build Kaggle feature matrix")
    p.add_argument('--config', default='configs/kaggle/war_mode.yaml')
    p.add_argument('--input', default=None)
    p.add_argument('--items', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--catalog', default=None)
    p.add_argument('--summary', default=None)
    return p.parse_args()


def main():
    args=parse_args(); cfg=load_kaggle_config(args.config)
    paths=cfg['paths']; reports=cfg.get('reports', {})
    df=read_table(args.input or paths.get('train_model_input') or paths.get('train_merged'))
    items=read_table(args.items or paths['items'])
    fp_cfg=FeaturePipelineConfig.from_dict(cfg.get('feature_engineering', {}))
    pipe=FeaturePipeline(fp_cfg, items=items)
    result=pipe.transform(df)
    out=args.output or paths.get('train_features', 'data/processed/train_features.parquet')
    write_table(result.features, out)
    write_feature_reports(
        result,
        args.catalog or reports.get('feature_catalog', 'reports/features/feature_catalog.json'),
        args.summary or reports.get('feature_summary', 'reports/features/feature_summary.md'),
        result.features.head(5),
    )
    print(f"[OK] features={result.features.shape} output={out}")

if __name__ == '__main__':
    main()
