#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.experiments.experiment_registry import load_registry
from src_kaggle.experiments.model_comparison import compare_oof_models


def main():
    p=argparse.ArgumentParser(description='Compare OOF experiments')
    p.add_argument('--registry', default='reports/experiments/experiment_registry.csv')
    p.add_argument('--experiments', required=False, help='comma-separated experiment names; default top all registry rows')
    p.add_argument('--out-dir', default='reports/experiments/comparison')
    args=p.parse_args()
    reg=load_registry(args.registry)
    if reg.empty:
        raise SystemExit('registry is empty')
    if args.experiments:
        names=[x.strip() for x in args.experiments.split(',')]
        reg=reg[reg['experiment_name'].isin(names)]
    paths={row['experiment_name']: row['oof_path'] for _,row in reg.iterrows() if Path(str(row['oof_path'])).exists()}
    if len(paths)<1:
        raise SystemExit('no OOF paths found')
    print(compare_oof_models(paths,args.out_dir))
if __name__=='__main__': main()
