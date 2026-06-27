#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.data.io import read_table
from src_kaggle.training.train_tabular import train_tabular_oof
from src_kaggle.utils.config import load_kaggle_config

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/kaggle/war_mode.yaml'); p.add_argument('--train',default=None); p.add_argument('--items',default=None); a=p.parse_args()
 cfg=load_kaggle_config(a.config); df=read_table(a.train or cfg['paths'].get('train_model_input')); items=read_table(a.items or cfg['paths']['items'])
 res=train_tabular_oof(df, items, cfg.get('modeling',{}).get('tabular',{})); print(res)
if __name__=='__main__': main()
