#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_kaggle.data.io import read_table
from src_kaggle.pseudo_labeling.pseudo_labeler import generate_pseudo_labels, write_pseudo_outputs
from src_kaggle.utils.config import load_kaggle_config

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/kaggle/war_mode.yaml'); p.add_argument('--candidates',required=True); p.add_argument('--output',default='data/processed/pseudo_labels.parquet'); p.add_argument('--report-dir',default='reports/pseudo_labeling'); a=p.parse_args()
 cfg=load_kaggle_config(a.config); cand=read_table(a.candidates); pseudo,rep=generate_pseudo_labels(cand,cfg.get('pseudo_labeling_controlled',{})); print(json.dumps(write_pseudo_outputs(pseudo,rep,a.output,a.report_dir),indent=2))
if __name__=='__main__': main()
