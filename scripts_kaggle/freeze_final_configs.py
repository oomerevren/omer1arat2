#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.finalization.config_freezer import freeze_family_configs

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config-dir', default='configs/kaggle/final'); p.add_argument('--artifact-root', default='artifacts/final_submissions'); p.add_argument('--reports-dir', default='reports/final'); args=p.parse_args()
    df=freeze_family_configs(args.config_dir, args.artifact_root, args.reports_dir)
    print(f"[OK] frozen configs={len(df)} -> {args.reports_dir}/final_config_freeze_index.csv")
