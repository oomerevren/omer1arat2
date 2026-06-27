from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig


def test_dense_retrieval_features_exist():
    items = pd.DataFrame([
        {SCHEMA.item_id:'i1', SCHEMA.title:'erkek deri ceket siyah', SCHEMA.category:'Giyim>Ceket', SCHEMA.brand:'A', SCHEMA.gender:'erkek', SCHEMA.age_group:'yetişkin', SCHEMA.attributes:'renk:siyah, materyal:deri'},
        {SCHEMA.item_id:'i2', SCHEMA.title:'kadın kırmızı elbise', SCHEMA.category:'Giyim>Elbise', SCHEMA.brand:'B', SCHEMA.gender:'kadın', SCHEMA.age_group:'yetişkin', SCHEMA.attributes:'renk:kırmızı'},
    ])
    pairs = pd.DataFrame([{SCHEMA.term_id:'t1', SCHEMA.item_id:'i1', SCHEMA.query:'erkek deri ceket', SCHEMA.title:'erkek deri ceket siyah', SCHEMA.category:'Giyim>Ceket', SCHEMA.brand:'A', SCHEMA.gender:'erkek', SCHEMA.age_group:'yetişkin', SCHEMA.attributes:'renk:siyah, materyal:deri'}])
    fp = FeaturePipeline(FeaturePipelineConfig(retrieval_top_k=2), items=items)
    res = fp.transform(pairs)
    for c in ['retrieval_dense_score_real','retrieval_dense_rank_real','retrieval_dense_only_hit_flag','retrieval_bm25_dense_overlap_flag','retrieval_hybrid_consensus_flag']:
        assert c in res.features.columns


if __name__ == '__main__':
    for name, obj in list(globals().items()):
        if name.startswith('test_') and callable(obj):
            obj()
    print('test_dense_feature_integration ok')
