from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.retrieval.hybrid_retriever import HybridRetriever
from src_kaggle.data.schema import SCHEMA


def sample_items():
    return pd.DataFrame({
        'item_id':[1,2,3,4,5,6],
        'title':['Nike beyaz kadın sneaker','Erkek suni deri ceket','Bebek pamuk zıbın','Unisex okul çantası','Koşu ayakkabısı 42','Siyah oversize sweatshirt'],
        'category':['sneaker','ceket','zıbın','okul çantası','ayakkabı','sweatshirt'],
        'brand':['Nike','Derimod','MiniCo','BagCo','RunCo','SweatCo'],
        'gender':['Kadın','Erkek','Bebek','Unisex','Erkek','Unisex'],
        'age_group':['Yetişkin','Yetişkin','Bebek','Çocuk','Yetişkin','Yetişkin'],
        'attributes':['renk: beyaz, stil: spor','renk: siyah, materyal: suni deri','materyal: pamuk','renk: siyah','numara: 42','renk: siyah, stil: oversize'],
    })


def main():
    r = HybridRetriever.build(sample_items(), dense_model_name='tfidf_svd_fallback', seed=1)
    bm25 = r.lexical_nearest_pool('nike beyaz kadın sneaker', 3)
    dense = r.dense_nearest_pool('erkek deri ceket', 3)
    hybrid = r.hybrid_search('siyah oversize sweatshirt', 3)
    cat = r.same_category_pool({'ceket'}, 2)
    brand = r.same_brand_pool({'Nike'}, 2)
    attr = r.attribute_similar_pool('siyah ceket', 3)
    far = r.random_far_pool('nike beyaz kadın sneaker', 2)
    assert not bm25.empty and bm25.iloc[0][SCHEMA.item_id] == 1
    assert not dense.empty
    assert not hybrid.empty
    assert not cat.empty and set(cat[SCHEMA.category]) == {'ceket'}
    assert not brand.empty and set(brand[SCHEMA.brand]) == {'Nike'}
    assert not attr.empty
    assert len(far) <= 2
    print('retrieval examples ok')

if __name__ == '__main__':
    main()
