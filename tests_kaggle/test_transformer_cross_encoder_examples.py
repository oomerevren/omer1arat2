from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_kaggle.models.transformer_dataset import build_pair_texts

def main():
    df=pd.DataFrame({'query':['a b'],'title':['a c'],'category':['cat'],'brand':['br'],'normalized_attribute_text':['color: black'],'gender':['unisex'],'age_group':['adult'],'label':[1]})
    texts=build_pair_texts(df,'full_v2')
    assert len(texts)==1 and '[QUERY]' in texts[0] and '[ATTR]' in texts[0]
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained('distilbert-base-uncased')
        enc=tok(texts, truncation=True, max_length=32, padding=True)
        assert 'input_ids' in enc
        print('transformer input prep ok')
    except Exception:
        print('transformers not available or model not cached; input text prep ok')
if __name__=='__main__': main()
