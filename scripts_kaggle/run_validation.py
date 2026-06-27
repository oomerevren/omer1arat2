#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_kaggle.data.io import read_table
from src_kaggle.utils.config import load_kaggle_config
from src_kaggle.validation.private_lb_simulator import run_validation

def main():
    p=argparse.ArgumentParser(description='Private-LB oriented validation')
    p.add_argument('--config',default='configs/kaggle/war_mode.yaml'); p.add_argument('--train',default=None); p.add_argument('--items',default=None); p.add_argument('--model-type',choices=['tabular','cross_encoder'],default=None); p.add_argument('--splitter',choices=['term_group','query_group','tail_aware'],default=None); p.add_argument('--out-dir',default='reports/validation')
    a=p.parse_args(); cfg=load_kaggle_config(a.config); val_cfg=dict(cfg.get('validation_framework',{}))
    if a.model_type: val_cfg['model_type']=a.model_type
    if a.splitter: val_cfg['splitter']=a.splitter
    if 'model' not in val_cfg: val_cfg['model']=cfg.get('modeling',{}).get(val_cfg.get('model_type','tabular'),{})
    if 'features' not in val_cfg: val_cfg['features']=cfg.get('feature_engineering',{})
    df=read_table(a.train or cfg['paths'].get('train_model_input') or cfg['paths'].get('train_merged'))
    items=read_table(a.items or cfg['paths']['items'])
    summary,_=run_validation(df,items,val_cfg,a.out_dir); print(summary)
if __name__=='__main__': main()
