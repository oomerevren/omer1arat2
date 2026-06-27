from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_kaggle.models.pair_text_builder import build_pair_text, describe_pair_text_formats

def main():
    row=pd.Series({'query':'nike beyaz kadın sneaker','title':'nike air sneaker','category':'ayakkabı/sneaker','brand':'nike','normalized_attribute_text':'color: white ; style: sport','gender':'Kadın','age_group':'Yetişkin','detected_color_candidates':'white'})
    for fmt in describe_pair_text_formats():
        txt=build_pair_text(row,fmt); assert '[QUERY]' in txt and '[SEP]' in txt
    assert '[INTENT]' in build_pair_text(row,'full_v2')
    print('pair text formats ok')
if __name__=='__main__': main()
