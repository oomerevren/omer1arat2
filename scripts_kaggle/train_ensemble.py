#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.training.train_ensemble import train_weighted_ensemble
from src_kaggle.utils.config import load_kaggle_config

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/kaggle/war_mode.yaml'); p.add_argument('--tabular-oof',default='artifacts/oof/tabular_oof.csv'); p.add_argument('--cross-oof',default='artifacts/oof/cross_encoder_oof.csv'); a=p.parse_args()
 cfg=load_kaggle_config(a.config); print(train_weighted_ensemble({'tabular':a.tabular_oof,'cross_encoder':a.cross_oof}, cfg.get('modeling',{}).get('ensemble',{})))
if __name__=='__main__': main()
