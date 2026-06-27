from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import tempfile
import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.retrieval.dense_retriever import DenseRetriever
from src_kaggle.retrieval.dense_index_store import save_dense_store, load_dense_store


def _items():
    return pd.DataFrame([
        {SCHEMA.item_id:'i1', SCHEMA.title:'erkek deri ceket siyah', SCHEMA.category:'Giyim>Ceket', SCHEMA.brand:'A', SCHEMA.gender:'erkek', SCHEMA.age_group:'yetişkin', SCHEMA.attributes:'renk:siyah, materyal:deri'},
        {SCHEMA.item_id:'i2', SCHEMA.title:'kadın elbise kırmızı', SCHEMA.category:'Giyim>Elbise', SCHEMA.brand:'B', SCHEMA.gender:'kadın', SCHEMA.age_group:'yetişkin', SCHEMA.attributes:'renk:kırmızı'},
        {SCHEMA.item_id:'i3', SCHEMA.title:'bebek zıbın pamuk', SCHEMA.category:'Bebek>Zıbın', SCHEMA.brand:'C', SCHEMA.gender:'unisex', SCHEMA.age_group:'bebek', SCHEMA.attributes:'materyal:pamuk'},
    ])


def test_fallback_dense_build_load_search_stable():
    r = DenseRetriever.fit(_items(), backend='fallback_dense', model_name='tfidf_svd_fallback', item_text_version='dense_v1')
    out = r.search('erkek deri ceket', top_k=2)
    assert len(out) == 2
    assert {'dense_score','dense_rank','dense_backend'}.issubset(out.columns)
    with tempfile.TemporaryDirectory() as d:
        path = save_dense_store(r, d)
        r2 = load_dense_store(path)
        out2 = r2.search('erkek deri ceket', top_k=2)
    assert out[SCHEMA.item_id].tolist() == out2[SCHEMA.item_id].tolist()
    assert r2.metadata['semantic_backend_active'] is False


if __name__ == '__main__':
    for name, obj in list(globals().items()):
        if name.startswith('test_') and callable(obj):
            obj()
    print('test_real_dense_retrieval_examples ok')
