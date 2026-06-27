#!/usr/bin/env python
"""Rebuild ablation summary/recommendation from master table."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.ablation.ablation_reporting import finalize_master, write_ablation_reports


def main():
    p=argparse.ArgumentParser(); p.add_argument('--master', default='reports/ablation/ablation_master_table.csv'); p.add_argument('--out-dir', default='reports/ablation'); args=p.parse_args()
    df=pd.read_csv(args.master)
    master=finalize_master(df.to_dict('records'))
    write_ablation_reports(master, args.out_dir)
    print(f"[OK] summaries rebuilt from {args.master}")
if __name__=='__main__': main()
