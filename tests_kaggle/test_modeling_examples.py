from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.data.attribute_parser import add_attribute_features
from src_kaggle.data.pair_builder import build_full_item_text
from src_kaggle.features.query_intent import add_query_intent_features, build_query_intent_resources
from src_kaggle.models.tabular_model import TabularModel
from src_kaggle.models.cross_encoder_model import CrossEncoderModel
from src_kaggle.features.feature_pipeline import FeaturePipeline
from src_kaggle.models.ensemble import WeightedAverageEnsemble
from src_kaggle.models.oof import build_oof_frame


def data():
    df=pd.DataFrame({
        'id': range(12), 'term_id':[1,1,2,2,3,3,4,4,5,5,6,6], 'item_id':range(12),
        'query':['nike beyaz kadın sneaker','nike beyaz kadın sneaker','erkek deri ceket','erkek deri ceket','bebek zıbın','bebek zıbın','unisex okul çantası','unisex okul çantası','42 numara koşu ayakkabısı','42 numara koşu ayakkabısı','siyah oversize sweatshirt','siyah oversize sweatshirt'],
        'title':['Nike beyaz kadın sneaker','Nike erkek şort','Erkek deri ceket','Kadın elbise','Bebek zıbın','Erkek ceket','Unisex okul çantası','Bebek zıbın','Koşu ayakkabısı 42','Kırmızı elbise','Siyah oversize sweatshirt','Beyaz sneaker'],
        'category':['sneaker','şort','ceket','elbise','zıbın','ceket','okul çantası','zıbın','ayakkabı','elbise','sweatshirt','sneaker'],
        'brand':['Nike','Nike','Derimod','DressCo','MiniCo','Derimod','BagCo','MiniCo','RunCo','DressCo','SweatCo','Adidas'],
        'gender':['Kadın','Erkek','Erkek','Kadın','Bebek','Erkek','Unisex','Bebek','Erkek','Kadın','Unisex','Kadın'],
        'age_group':['Yetişkin','Yetişkin','Yetişkin','Yetişkin','Bebek','Yetişkin','Çocuk','Bebek','Yetişkin','Yetişkin','Yetişkin','Yetişkin'],
        'attributes':['renk: beyaz','renk: siyah','materyal: deri','renk: kırmızı','materyal: pamuk','materyal: deri','renk: siyah','materyal: pamuk','numara: 42','renk: kırmızı','renk: siyah, stil: oversize','renk: beyaz'],
        'label':[1,0,1,0,1,0,1,0,1,0,1,0]
    })
    df=build_full_item_text(add_attribute_features(df)); df=add_query_intent_features(df, build_query_intent_resources(df)); return df

def main():
    df=data(); fp=FeaturePipeline(items=df); X=fp.transform(df).features; y=df['label']
    tm=TabularModel(seed=1).fit(X,y); pt=tm.predict_proba(X); assert len(pt)==len(df)
    ce=CrossEncoderModel(seed=1).fit(df,y); pc=ce.predict_proba(df); assert len(pc)==len(df)
    o1=build_oof_frame(df, pt, 'tabular'); o2=build_oof_frame(df, pc, 'cross_encoder')
    ens=WeightedAverageEnsemble({'tabular':0.5,'cross_encoder':0.5}); rep=ens.fit_oof({'tabular':o1,'cross_encoder':o2}); assert 'macro_f1' in rep
    print('modeling examples ok', rep)
if __name__=='__main__': main()
