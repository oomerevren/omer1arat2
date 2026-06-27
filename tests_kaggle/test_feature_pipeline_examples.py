from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.features.feature_pipeline import FeaturePipeline


def items():
    return pd.DataFrame({
        'item_id':[1,2,3,4,5,6],
        'title':['Nike beyaz kadın sneaker','Erkek suni deri ceket','Bebek pamuk zıbın','Unisex okul çantası','Koşu ayakkabısı 42','Siyah oversize sweatshirt'],
        'category':['sneaker','ceket','zıbın','okul çantası','ayakkabı','sweatshirt'],
        'brand':['Nike','Derimod','MiniCo','BagCo','RunCo','SweatCo'],
        'gender':['Kadın','Erkek','Bebek','Unisex','Erkek','Unisex'],
        'age_group':['Yetişkin','Yetişkin','Bebek','Çocuk','Yetişkin','Yetişkin'],
        'attributes':['renk: beyaz, stil: spor','renk: siyah, materyal: suni deri','materyal: pamuk','renk: siyah','numara: 42','renk: siyah, stil: oversize'],
    })

def pairs():
    it=items()
    qs=['nike beyaz kadın sneaker','erkek deri ceket','bebek zıbın','unisex okul çantası','42 numara koşu ayakkabısı','siyah oversize sweatshirt']
    df=it.copy(); df['query']=qs; df['term_id']=range(10,16); df['label']=[1,1,1,1,1,1]; return df

def main():
    pipe=FeaturePipeline(items=items())
    res=pipe.transform(pairs())
    assert res.features.shape[1] >= 70
    for col in ['lex_token_overlap_ratio','brand_exact_match','attr_color_exact_match','gender_conflict_flag','retrieval_bm25_score','sem_query_title_cosine']:
        assert col in res.features.columns, col
    assert res.features.isna().sum().sum() == 0
    print('feature pipeline examples ok', res.features.shape)
    print(res.features[['lex_token_overlap_ratio','brand_exact_match','attr_color_exact_match','gender_conflict_flag','retrieval_bm25_score','sem_query_title_cosine']].head().to_string(index=False))
if __name__=='__main__': main()
