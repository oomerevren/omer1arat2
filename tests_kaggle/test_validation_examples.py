from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_kaggle.data.attribute_parser import add_attribute_features
from src_kaggle.data.pair_builder import build_full_item_text
from src_kaggle.features.query_intent import add_query_intent_features, build_query_intent_resources
from src_kaggle.validation.private_lb_simulator import run_validation

def data():
    base=[]
    queries=['nike beyaz kadın sneaker','erkek deri ceket','bebek zıbın','unisex okul çantası','42 numara koşu ayakkabısı','siyah oversize sweatshirt']
    pos_titles=['Nike beyaz kadın sneaker','Erkek deri ceket','Bebek zıbın','Unisex okul çantası','Koşu ayakkabısı 42','Siyah oversize sweatshirt']
    neg_titles=['Nike erkek şort','Kadın elbise','Erkek ceket','Bebek zıbın','Kırmızı elbise','Beyaz sneaker']
    for i,q in enumerate(queries):
        base.append({'id':i*2,'term_id':i,'item_id':i*2,'query':q,'title':pos_titles[i],'category':['sneaker','ceket','zıbın','okul çantası','ayakkabı','sweatshirt'][i],'brand':['Nike','Derimod','MiniCo','BagCo','RunCo','SweatCo'][i],'gender':['Kadın','Erkek','Bebek','Unisex','Erkek','Unisex'][i],'age_group':['Yetişkin','Yetişkin','Bebek','Çocuk','Yetişkin','Yetişkin'][i],'attributes':'renk: siyah, materyal: deri','label':1,'negative_type':'positive'})
        base.append({'id':i*2+1,'term_id':i,'item_id':i*2+1,'query':q,'title':neg_titles[i],'category':'other','brand':'Other','gender':'Kadın','age_group':'Yetişkin','attributes':'renk: kırmızı','label':0,'negative_type':'easy'})
    df=pd.DataFrame(base); df=build_full_item_text(add_attribute_features(df)); df=add_query_intent_features(df, build_query_intent_resources(df)); return df

def main():
    df=data(); cfg={'model_type':'tabular','splitter':'term_group','n_folds':3,'seeds':[1],'segment_min_rows':1,'model':{'model_type':'hist_gradient_boosting','params':{'max_iter':20}},'features':{'use_retrieval_features':False,'use_semantic_features':True}}
    summary,last=run_validation(df,df,cfg,'/tmp/validation_test')
    assert 'seed_stability' in summary and last['oof'].shape[0]==len(df)
    assert 'pred_best_threshold' in last['oof'].columns
    print('validation examples ok', summary['seed_results'])
if __name__=='__main__': main()
