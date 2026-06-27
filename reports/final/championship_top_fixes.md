# Championship Top Fixes

gap_id,component,recommended_fix
GAP-001,retrieval,Run build_retrieval_index + evaluate_dense_retrieval on official items/queries.
GAP-002,dense retrieval,Build real dense index with official items; compare dense_v1/v2 and BM25 overlap before final use.
GAP-003,negative mining,Use fold-aware negative builds for final validation; inspect uncertain/skipped rates.
GAP-004,semantic hard negatives,Keep dense hard negatives out of default unless ablation shows class0-safe lift.
GAP-005,tabular modeling,Run official OOF; keep tabular as baseline/family A backbone until CE proves lift.
