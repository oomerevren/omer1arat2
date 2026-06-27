#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.data.io import read_table
from src_kaggle.training.train_cross_encoder import train_cross_encoder_oof
from src_kaggle.utils.config import load_kaggle_config

def main():
    p=argparse.ArgumentParser(description='Train cross-encoder OOF (sklearn_text or real transformers)')
    p.add_argument('--config',default='configs/kaggle/war_mode.yaml'); p.add_argument('--train',default=None)
    p.add_argument('--backend',choices=['sklearn_text','transformers'],default=None)
    p.add_argument('--model-name',default=None); p.add_argument('--text-format',default=None); p.add_argument('--max-length',type=int,default=None)
    p.add_argument('--allow-cpu',action='store_true'); p.add_argument('--experiment-name',default='ce_transformer')
    a=p.parse_args(); cfg=load_kaggle_config(a.config); ce=dict(cfg.get('modeling',{}).get('cross_encoder',{}))
    if a.backend: ce['backend']=a.backend
    if a.model_name: ce['model_name']=a.model_name
    if a.text_format: ce['text_format_version']=a.text_format
    if a.max_length: ce['max_length']=a.max_length
    if a.allow_cpu: ce['allow_cpu']=True
    df=read_table(a.train or cfg['paths'].get('train_model_input'))
    res=train_cross_encoder_oof(df,ce,experiment_name=a.experiment_name); print(res)
if __name__=='__main__': main()
