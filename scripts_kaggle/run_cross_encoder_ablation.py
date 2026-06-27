#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.data.io import read_table
from src_kaggle.training.train_cross_encoder import train_cross_encoder_oof
from src_kaggle.utils.config import load_kaggle_config

def main():
    p=argparse.ArgumentParser(description='Run cross-encoder text/backbone ablation')
    p.add_argument('--config',default='configs/kaggle/war_mode.yaml'); p.add_argument('--train',default=None)
    p.add_argument('--backbones',default=None,help='comma separated'); p.add_argument('--text-formats',default='query_title,query_title_category,full_v1,full_v2')
    p.add_argument('--max-lengths',default='128,256'); p.add_argument('--backend',default='transformers'); p.add_argument('--allow-cpu',action='store_true')
    a=p.parse_args(); cfg=load_kaggle_config(a.config); base=dict(cfg.get('modeling',{}).get('cross_encoder',{})); df=read_table(a.train or cfg['paths'].get('train_model_input'))
    backbones=[x.strip() for x in (a.backbones or base.get('model_name','dbmdz/distilbert-base-turkish-cased')).split(',')]
    formats=[x.strip() for x in a.text_formats.split(',')]; lengths=[int(x) for x in a.max_lengths.split(',')]
    for b in backbones:
        for fmt in formats:
            for ml in lengths:
                ce={**base,'backend':a.backend,'model_name':b,'text_format_version':fmt,'max_length':ml}
                if a.allow_cpu: ce['allow_cpu']=True
                exp=f"ce_{a.backend}_{b.split('/')[-1]}_{fmt}_ml{ml}"
                print('[RUN]',exp); train_cross_encoder_oof(df,ce,experiment_name=exp)
if __name__=='__main__': main()
