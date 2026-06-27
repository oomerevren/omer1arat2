#!/usr/bin/env python
"""Run the controlled Kaggle ablation suite."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.utils.config import load_kaggle_config
from src_kaggle.ablation.ablation_runner import AblationRunner
from src_kaggle.ablation.ablation_specs import all_ablation_specs


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation suite")
    p.add_argument('--config', default='configs/kaggle/war_mode.yaml')
    p.add_argument('--category', action='append', choices=['feature','negative','retrieval','model','threshold','dense'])
    p.add_argument('--max-runs', type=int, default=None)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--allow-transformer', action='store_true')
    p.add_argument('--allow-real-dense', action='store_true')
    p.add_argument('--out-dir', default='reports/ablation')
    return p.parse_args()


def main():
    args = parse_args(); cfg = load_kaggle_config(args.config)
    runner = AblationRunner(cfg, out_dir=args.out_dir, max_runs=args.max_runs, smoke=args.smoke, allow_transformer=args.allow_transformer, allow_real_dense=args.allow_real_dense)
    master = runner.run(all_ablation_specs(), categories=set(args.category) if args.category else None)
    print(f"[OK] ablation rows={len(master)} -> {args.out_dir}/ablation_master_table.csv")
    print(master[['ablation_id','category','status','OOF macro-F1','risk_flag']].head(20).to_string(index=False))
if __name__ == '__main__': main()
