from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src_kaggle.data.schema import SCHEMA
from src_kaggle.data.negative_mining import NegativeMiningConfig, mine_negatives


def test_dense_hard_negative_metadata_and_uncertain():
    items = pd.DataFrame([
        {SCHEMA.item_id:'p1', SCHEMA.title:'erkek deri ceket siyah', SCHEMA.category:'Giyim>Ceket', SCHEMA.brand:'A', SCHEMA.gender:'erkek', SCHEMA.age_group:'yetişkin', SCHEMA.attributes:'renk:siyah, materyal:deri'},
        {SCHEMA.item_id:'n1', SCHEMA.title:'erkek deri mont kahverengi', SCHEMA.category:'Giyim>Ceket', SCHEMA.brand:'A', SCHEMA.gender:'erkek', SCHEMA.age_group:'yetişkin', SCHEMA.attributes:'renk:kahverengi, materyal:deri'},
        {SCHEMA.item_id:'n2', SCHEMA.title:'kadın deri ceket siyah', SCHEMA.category:'Giyim>Ceket', SCHEMA.brand:'A', SCHEMA.gender:'kadın', SCHEMA.age_group:'yetişkin', SCHEMA.attributes:'renk:siyah, materyal:deri'},
        {SCHEMA.item_id:'x', SCHEMA.title:'bebek pamuk zıbın', SCHEMA.category:'Bebek>Zıbın', SCHEMA.brand:'C', SCHEMA.gender:'unisex', SCHEMA.age_group:'bebek', SCHEMA.attributes:'materyal:pamuk'},
    ])
    pos = pd.DataFrame([{SCHEMA.id:1, SCHEMA.term_id:'t1', SCHEMA.item_id:'p1', SCHEMA.query:'erkek deri ceket', SCHEMA.label:1, SCHEMA.title:'erkek deri ceket siyah', SCHEMA.category:'Giyim>Ceket', SCHEMA.brand:'A'}])
    cfg = NegativeMiningConfig(enabled=True, dense_negatives_per_positive=2, use_dense_pool=True, easy_negatives_per_positive=0, same_category_negatives_per_positive=0, same_brand_negatives_per_positive=0, lexical_negatives_per_positive=0, attribute_conflict_negatives_per_positive=0, retrieval_cfg={'enabled_retrievers':['dense'], 'dense': {'enabled': True, 'backend':'fallback_dense', 'model_name':'tfidf_svd_fallback', 'item_text_version':'dense_v1'}})
    neg, unc = mine_negatives(pos, items, cfg)
    assert (len(neg) + len(unc)) > 0
    allc = pd.concat([neg, unc], ignore_index=True, sort=False)
    assert {'dense_score','dense_negative_subtype','safety_status','dense_backend'}.issubset(allc.columns)
    assert allc['source_pool'].eq('dense_nearest_pool').any()


if __name__ == '__main__':
    for name, obj in list(globals().items()):
        if name.startswith('test_') and callable(obj):
            obj()
    print('test_dense_hard_negative_examples ok')
