#!/usr/bin/env python
"""Evaluate BM25 vs dense retrieval complementarity."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.data.io import read_table
from src_kaggle.retrieval.retrieval_index import build_retrieval_index
from src_kaggle.retrieval.retrieval_evaluator import evaluate_retrieval, write_retrieval_evaluation
from src_kaggle.retrieval.semantic_confuser_analysis import write_semantic_confuser_report
from src_kaggle.utils.config import load_kaggle_config


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/kaggle/war_mode.yaml'); p.add_argument('--top-k', type=int, default=None); return p.parse_args()

def main():
    args=parse_args(); cfg=load_kaggle_config(args.config); paths=cfg['paths']; reports=cfg.get('reports',{}); retrieval_cfg=cfg.get('retrieval',{})
    items=read_table(paths['items']); train=read_table(paths.get('train_merged') or paths['training_pairs'])
    index=build_retrieval_index(items, retrieval_cfg)
    top_k=args.top_k or int((retrieval_cfg.get('dense') or {}).get('top_k', retrieval_cfg.get('dense_top_k', 50)))
    comp, report=evaluate_retrieval(train, index, top_k=top_k)
    write_retrieval_evaluation(comp, report, comparison_csv=reports.get('dense_vs_bm25_comparison','reports/retrieval/dense_vs_bm25_comparison.csv'), segment_json=reports.get('query_segment_retrieval_comparison','reports/retrieval/query_segment_retrieval_comparison.json'))
    # Placeholder confuser report from retrieval behavior if negatives not yet built.
    write_semantic_confuser_report(__import__('pandas').DataFrame(), __import__('pandas').DataFrame(), reports.get('semantic_confuser_examples','reports/retrieval/semantic_confuser_examples.md'))
    print(f"[OK] dense evaluation rows={len(comp)} backend={report['dense_backend']} semantic_active={report['semantic_backend_active']}")
    if not report['semantic_backend_active']: print('[WARN] fallback_dense aktif; gerçek semantic retrieval raporu değildir.')
if __name__=='__main__': main()
