#!/usr/bin/env python
"""Safe Kaggle submission generator. Output is strictly id,prediction."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.data.io import read_table
from src_kaggle.models.baseline import TfidfLogRegPairClassifier
from src_kaggle.submission.submission_builder import build_submission_from_proba, build_submission_from_predictions, save_safe_submission, append_submission_registry
from src_kaggle.utils.config import load_kaggle_config

def parse_args():
    p=argparse.ArgumentParser(description='Make validated Kaggle submission')
    p.add_argument('--config',default='configs/kaggle/war_mode.yaml')
    p.add_argument('--test-pairs',default=None)
    p.add_argument('--submission-pairs',default=None)
    p.add_argument('--proba-file',default=None,help='CSV with id,proba or row-order proba')
    p.add_argument('--pred-file',default=None,help='CSV with id,prediction or row-order prediction')
    p.add_argument('--model',default=None)
    p.add_argument('--metrics',default=None)
    p.add_argument('--output',default=None)
    p.add_argument('--threshold',type=float,default=None)
    p.add_argument('--experiment-id',default='')
    p.add_argument('--models-used',default='')
    p.add_argument('--previous-submission',default=None)
    p.add_argument('--public-lb-score',default='')
    p.add_argument('--note',default='')
    return p.parse_args()

def _aligned_values(df, ref, value_col):
    if 'id' in df.columns:
        return df.set_index('id').reindex(ref['id'])[value_col].values
    return df[value_col].values if value_col in df.columns else df.iloc[:,0].values

def main():
    args=parse_args(); cfg=load_kaggle_config(args.config); paths=cfg['paths']; reports=cfg.get('reports',{})
    test_pairs=read_table(args.test_pairs or paths.get('test_model_input') or paths.get('test_merged'))
    reference=read_table(args.submission_pairs or paths['submission_pairs'])[["id"]]
    threshold=args.threshold
    metrics_path=args.metrics or paths.get('metrics')
    if threshold is None and metrics_path and Path(metrics_path).exists(): threshold=float(json.loads(Path(metrics_path).read_text()).get('threshold',0.5))
    if threshold is None: threshold=float(cfg.get('model',{}).get('binary_threshold_default',0.5))
    if args.pred_file:
        pf=read_table(args.pred_file); sub=build_submission_from_predictions(reference, _aligned_values(pf, reference, 'prediction'))
    else:
        if args.proba_file:
            pf=read_table(args.proba_file); proba=_aligned_values(pf, reference, 'proba')
        else:
            model=TfidfLogRegPairClassifier.load(args.model or paths['model']); proba=model.predict_proba(test_pairs)
        sub=build_submission_from_proba(reference, proba, threshold)
    prev=read_table(args.previous_submission) if args.previous_submission else None
    output=args.output or paths['submission']; report=reports.get('submission_validation','reports/submissions/submission_validation_report.json')
    result=save_safe_submission(sub, reference, output, report_path=report, previous_submission=prev)
    append_submission_registry({'experiment_id':args.experiment_id,'models':args.models_used,'threshold':threshold,'positive_rate':result['positive_rate'],'file_path':output,'validation_report':report,'public_lb_score':args.public_lb_score,'note':args.note}, reports.get('submission_registry','reports/submissions/submission_registry.csv'))
    print(f"[OK] safe submission saved: {output} positive_rate={result['positive_rate']:.4f}")
if __name__=='__main__': main()
