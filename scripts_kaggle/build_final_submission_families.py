#!/usr/bin/env python
"""Build final candidate pool, blends and three strategic submission families."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.utils.config import load_kaggle_config
from src_kaggle.final.candidate_selector import select_final_candidates, write_candidate_pool
from src_kaggle.final.ensemble_optimizer import optimize_blends, write_blend_comparison
from src_kaggle.final.submission_family_builder import choose_family_models, materialize_family_artifacts
from src_kaggle.final.family_risk_profiler import build_family_segment_scores, build_family_risk_flags
from src_kaggle.final.final_recommender import write_final_recommendation


def parse_args():
    p=argparse.ArgumentParser(description='Build final submission families')
    p.add_argument('--config', default='configs/kaggle/war_mode.yaml')
    p.add_argument('--leaderboard', default='reports/leaderboard/oof_public_correlation.csv')
    p.add_argument('--ablation', default='reports/ablation/ablation_master_table.csv')
    p.add_argument('--out-dir', default='reports/final')
    p.add_argument('--artifact-root', default='artifacts/final_submissions')
    p.add_argument('--configs-root', default='configs/kaggle/final')
    return p.parse_args()


def main():
    args=parse_args(); cfg=load_kaggle_config(args.config); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pool=select_final_candidates(args.leaderboard, args.ablation)
    write_candidate_pool(pool, out/'final_candidate_pool.csv')
    blend_table, blend_report=optimize_blends(pool)
    write_blend_comparison(blend_table, blend_report, out/'final_blend_comparison.json', out/'final_blend_comparison.md')
    families=choose_family_models(pool, blend_table)
    sub_pairs = cfg.get('paths',{}).get('submission_pairs')
    fam_df=materialize_family_artifacts(families, pool, submission_pairs_path=sub_pairs, artifact_root=args.artifact_root, configs_root=args.configs_root, registry_path=cfg.get('reports',{}).get('submission_registry','reports/submissions/submission_registry.csv'))
    fam_df.to_csv(out/'final_submission_families.csv', index=False)
    seg=build_family_segment_scores(fam_df, out/'final_family_segment_scores.csv')
    risks=build_family_risk_flags(fam_df, out/'final_family_risk_flags.csv')
    write_final_recommendation(fam_df, blend_report, out/'final_recommendation.md')
    print(f"[OK] final candidate pool rows={len(pool)}")
    print(f"[OK] final families rows={len(fam_df)}")
    print(f"[OK] reports={out}")
    print(fam_df[['family_name','public_private_risk_label','status','threshold']].to_string(index=False))

if __name__=='__main__': main()
