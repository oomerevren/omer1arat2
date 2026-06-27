#!/usr/bin/env python
"""Build/persist hybrid or dense-only retrieval index and example reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.data.io import read_table
from src_kaggle.data.schema import SCHEMA
from src_kaggle.retrieval.retrieval_index import build_retrieval_index, save_retrieval_index
from src_kaggle.retrieval.embedding_builder import build_and_optionally_persist_dense, dense_cfg_from_retrieval_cfg
from src_kaggle.utils.config import load_kaggle_config

EXAMPLE_QUERIES = ["nike beyaz kadın sneaker", "erkek deri ceket", "bebek zıbın", "unisex okul çantası", "42 numara koşu ayakkabısı", "siyah oversize sweatshirt"]


def parse_args():
    p = argparse.ArgumentParser(description="Build Kaggle retrieval index")
    p.add_argument("--config", default="configs/kaggle/war_mode.yaml")
    p.add_argument("--items", default=None)
    p.add_argument("--index-out", default=None)
    p.add_argument("--report-json", default=None)
    p.add_argument("--examples-md", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "dense_only"], default="hybrid")
    return p.parse_args()


def _mini_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df is None or df.empty:
        return "_no results_\n"
    cols = [c for c in cols if c in df.columns]
    return df[cols].head(10).to_markdown(index=False) + "\n"


def main() -> None:
    args = parse_args(); cfg = load_kaggle_config(args.config)
    retrieval_cfg = cfg.get("retrieval", {}); paths = cfg["paths"]; reports = cfg.get("reports", {})
    items = read_table(args.items or paths["items"])

    dense_artifact = None
    if args.mode == "dense_only":
        dense, dense_artifact = build_and_optionally_persist_dense(items, retrieval_cfg)
        from src_kaggle.retrieval.hybrid_retriever import HybridRetriever
        index = HybridRetriever(items=dense.items if dense is not None else items, bm25=None, dense=dense, seed=int(retrieval_cfg.get("seed", 42)), dense_text_version=dense_cfg_from_retrieval_cfg(retrieval_cfg).get("item_text_version", "dense_v1"))
    else:
        index = build_retrieval_index(items, retrieval_cfg)

    index_path = args.index_out or retrieval_cfg.get("index_path") or paths.get("retrieval_index", "artifacts/retrieval/hybrid_retriever.pkl")
    if retrieval_cfg.get("persist_indices", True): save_retrieval_index(index, index_path)

    empty_rate = float(index.items[SCHEMA.retrieval_text].fillna("").astype(str).str.strip().eq("").mean())
    dense_meta = getattr(index.dense, "metadata", {}) if index.dense is not None else {}
    report = {"index_path": index_path, "dense_artifact_dir": dense_artifact, "total_items": int(len(index.items)), "item_text_empty_rate": empty_rate, "enabled_retrievers": retrieval_cfg.get("enabled_retrievers", ["bm25", "dense"]), "mode": args.mode, "dense": dense_meta, "bm25_available": index.bm25 is not None, "dense_available": index.dense is not None, "dense_backend": getattr(index.dense, "backend", "none"), "semantic_backend_active": bool(index.dense and index.dense.is_real_dense), "examples": {}}
    if index.dense is not None and not index.dense.is_real_dense:
        report["warning"] = "Dense backend fallback_dense; real semantic hard negative iddiası kurulmaz."

    md = ["# Retrieval Examples", "", f"Dense backend: `{report['dense_backend']}` / semantic active: `{report['semantic_backend_active']}`", ""]
    for q in EXAMPLE_QUERIES:
        bm25 = index.lexical_nearest_pool(q, args.top_k)
        dense = index.dense_nearest_pool(q, args.top_k)
        hybrid = index.hybrid_search(q, top_k=args.top_k, bm25_k=args.top_k, dense_k=args.top_k)
        bm25_ids = set(bm25[SCHEMA.item_id].tolist()) if not bm25.empty else set(); dense_ids = set(dense[SCHEMA.item_id].tolist()) if not dense.empty else set()
        overlap = len(bm25_ids & dense_ids) / max(1, len(bm25_ids | dense_ids))
        report["examples"][q] = {"bm25_item_ids": list(map(str, bm25_ids)), "dense_item_ids": list(map(str, dense_ids)), "bm25_dense_overlap_jaccard": float(overlap)}
        md += [f"## {q}", "", "### BM25", _mini_table(bm25, [SCHEMA.item_id, SCHEMA.title, SCHEMA.category, SCHEMA.brand, "score"]), "", "### Dense", _mini_table(dense, [SCHEMA.item_id, SCHEMA.title, SCHEMA.category, SCHEMA.brand, "dense_score", "dense_rank"]), "", "### Hybrid", _mini_table(hybrid, [SCHEMA.item_id, SCHEMA.title, SCHEMA.category, SCHEMA.brand, "hybrid_score", "source"]), f"BM25/Dense overlap Jaccard: `{overlap:.3f}`", ""]

    report_json = Path(args.report_json or reports.get("retrieval_index_json", "reports/retrieval/index_report.json")); examples_md = Path(args.examples_md or reports.get("retrieval_examples_md", "reports/retrieval/retrieval_examples.md"))
    report_json.parent.mkdir(parents=True, exist_ok=True); examples_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"); examples_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] retrieval index items={len(index.items)} backend={report['dense_backend']} semantic_active={report['semantic_backend_active']} path={index_path}")
    if report.get("warning"): print(f"[WARN] {report['warning']}")

if __name__ == "__main__": main()
