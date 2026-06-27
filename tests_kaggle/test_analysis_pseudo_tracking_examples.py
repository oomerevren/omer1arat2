from __future__ import annotations
import sys, tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_kaggle.analysis.error_analysis import analyze_errors
from src_kaggle.pseudo_labeling.pseudo_labeler import generate_pseudo_labels
from src_kaggle.tracking.experiment_tracker import log_run

def main():
    oof=pd.DataFrame({'id':[1,2,3,4],'term_id':[1,1,2,2],'item_id':[10,11,12,13],'label':[1,0,1,0],'proba':[.2,.8,.9,.1],'pred_best_threshold':[0,1,1,0],'query':['nike ayakkabı','nike ayakkabı','erkek ceket','erkek ceket'],'title':['adidas sneaker','nike şort','erkek ceket','kadın elbise'],'brand_contradiction_flag':[1,0,0,0],'gender_conflict_flag':[0,0,0,1],'negative_type':['positive','same_brand','positive','easy']})
    res=analyze_errors(oof,0.5); assert res['summary']['total_errors']==2
    cand=pd.DataFrame({'id':[5,6],'proba':[.97,.02],'lex_token_overlap_ratio':[.5,0.0],'retrieval_bm25_score':[1,0],'attr_conflict_count':[0,1],'gender_conflict_flag':[0,1],'age_conflict_flag':[0,0],'category':['a','b']})
    pseudo,rep=generate_pseudo_labels(cand,{'mode':'dual','positive_threshold':.95,'negative_threshold':.05,'min_margin':.3}); assert len(pseudo)==2
    with tempfile.TemporaryDirectory() as d:
        df=log_run({'run_id':'r1','experiment_name':'e','oof_macro_f1':.7,'pseudo_label_count':2},Path(d)/'master.csv'); assert len(df)==1
    print('analysis pseudo tracking examples ok')
if __name__=='__main__': main()
