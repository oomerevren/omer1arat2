from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src_kaggle.validation.fold_runner import ValidationFoldRunner
from src_kaggle.validation.oof_manager import make_oof, write_oof
from src_kaggle.validation.segment_reports import segment_scores
from src_kaggle.validation.seed_stability import summarize_seed_results
from src_kaggle.validation.threshold_tuning import metrics_at_threshold


def run_validation(df, items=None, cfg=None, out_dir='reports/validation'):
    cfg=cfg or {}; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    seeds=cfg.get('seeds',[cfg.get('seed',42)]); seed_summaries=[]; last=None
    for seed in seeds:
        runner=ValidationFoldRunner(model_type=cfg.get('model_type','tabular'), model_cfg=cfg.get('model',{}), feature_cfg=cfg.get('features',{}), items=items, n_splits=int(cfg.get('n_folds',5)), seed=int(seed), splitter=cfg.get('splitter','term_group'))
        res=runner.run(df); oof=make_oof(df,res['proba'],res['folds'],res['best_threshold'],cfg.get('model_type','tabular'))
        prefix=f"seed_{seed}_" if len(seeds)>1 else ''
        write_oof(oof,out/f'{prefix}oof_predictions.csv')
        res['fold_scores'].to_csv(out/f'{prefix}fold_scores.csv',index=False)
        res['threshold_curve'].to_csv(out/f'{prefix}threshold_curve.csv',index=False)
        seg=segment_scores(oof,res['best_threshold'],min_rows=int(cfg.get('segment_min_rows',5))); seg.to_csv(out/f'{prefix}segment_scores.csv',index=False)
        m=metrics_at_threshold(oof['label'],oof['proba'],res['best_threshold'])
        seed_summaries.append({'seed':seed,'macro_f1':m['macro_f1'],'class0_f1':m['class0_f1'],'class1_f1':m['class1_f1'],'best_threshold':res['best_threshold']})
        last={'oof':oof,'result':res,'segments':seg,'metrics':m}
    stability=summarize_seed_results(seed_summaries)
    summary={'config':cfg,'seed_results':seed_summaries,'seed_stability':stability,'recommended_threshold': seed_summaries[-1]['best_threshold'] if seed_summaries else 0.5}
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    (out/'seed_stability.json').write_text(json.dumps(stability,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary,last
