#!/usr/bin/env python
"""Analyze OOF/public-LB correlation and private-LB defense risk."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.leaderboard.oof_public_linker import build_oof_public_table, append_public_lb_entry
from src_kaggle.leaderboard.risk_flagging import apply_risk_flags
from src_kaggle.leaderboard.correlation_analysis import safe_corr, sign_agreement
from src_kaggle.leaderboard.splitter_reliability import splitter_reliability, write_splitter_report
from src_kaggle.leaderboard.model_family_drift import model_family_drift, write_model_family_report
from src_kaggle.leaderboard.threshold_reliability import threshold_reliability, write_threshold_report
from src_kaggle.leaderboard.segment_risk import collect_segment_risk, write_segment_risk
from src_kaggle.leaderboard.private_lb_strategy import write_private_lb_flags, write_strategy_recommendation


def parse_args():
    p=argparse.ArgumentParser(description='OOF/public leaderboard intelligence')
    p.add_argument('--out-dir', default='reports/leaderboard')
    p.add_argument('--experiment-registry', default='reports/experiments/experiment_registry.csv')
    p.add_argument('--submission-registry', default='reports/submissions/submission_registry.csv')
    p.add_argument('--ablation-master', default='reports/ablation/ablation_master_table.csv')
    p.add_argument('--public-tracking', default='reports/leaderboard/public_lb_tracking_table.csv')
    p.add_argument('--add-public-entry', action='store_true')
    p.add_argument('--file-path', default='')
    p.add_argument('--public-lb-score', default='')
    p.add_argument('--experiment-id', default='')
    p.add_argument('--experiment-name', default='')
    p.add_argument('--notes', default='')
    p.add_argument('--chosen-family-label', default='')
    return p.parse_args()


def main():
    args=parse_args(); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.add_public_entry:
        append_public_lb_entry(file_path=args.file_path, public_lb_score=args.public_lb_score, experiment_id=args.experiment_id, experiment_name=args.experiment_name, notes=args.notes, chosen_family_label=args.chosen_family_label, tracking_path=args.public_tracking)
    table=build_oof_public_table(experiment_registry=args.experiment_registry, submission_registry=args.submission_registry, ablation_master=args.ablation_master, public_lb_tracking=args.public_tracking)
    table=apply_risk_flags(table)
    # stable column order requested by prompt
    cols=['experiment_id','experiment_name','submission_id','file_path','model_family','splitter','threshold','OOF macro-F1','class0 F1','class1 F1','public_lb_score','public_minus_oof','public_rank_delta','threshold_fragility','seed_std','risk_flag','strategic_status','notes']
    for c in cols:
        if c not in table.columns: table[c]=''
    table[cols].to_csv(out/'oof_public_correlation.csv', index=False)

    splitter_table, splitter_report=splitter_reliability(table); write_splitter_report(splitter_table, splitter_report, out)
    family_table, family_report=model_family_drift(table); write_model_family_report(family_table, family_report, out)
    threshold_table, threshold_report=threshold_reliability(table); write_threshold_report(threshold_table, threshold_report, out)
    segment_table=collect_segment_risk(table); write_segment_risk(segment_table, out)
    write_private_lb_flags(table, segment_table, out)
    write_strategy_recommendation(table, splitter_report, family_report, threshold_report, out)

    summary={
        'overall_correlation': safe_corr(table),
        'sign_agreement': sign_agreement(table),
        'n_rows': int(len(table)),
        'n_public_lb': int(table['public_lb_score'].notna().sum()) if 'public_lb_score' in table else 0,
        'decision_priority': ['OOF macro-F1','class0 F1','splitter reliability','threshold fragility','seed stability','segment collapse risk','model family drift','public LB'],
        'small_sample_warning': 'Do not infer private-LB truth from fewer than 3 public submissions.'
    }
    (out/'oof_public_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    md=['# OOF/Public Summary','',f"Rows: {summary['n_rows']}",f"Public LB points: {summary['n_public_lb']}",'',f"Correlation: {summary['overall_correlation']}",'',f"Sign agreement: {summary['sign_agreement']}",'','## Decision priority','']
    md += [f"{i+1}. {x}" for i,x in enumerate(summary['decision_priority'])]
    md += ['', summary['small_sample_warning']]
    (out/'oof_public_summary.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(f"[OK] leaderboard analysis rows={len(table)} public_points={summary['n_public_lb']} -> {out/'oof_public_correlation.csv'}")

if __name__=='__main__': main()
