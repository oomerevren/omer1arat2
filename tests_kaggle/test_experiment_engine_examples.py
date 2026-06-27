from __future__ import annotations
import sys
from pathlib import Path
import tempfile
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_kaggle.validation.threshold_optimizer import optimize_oof_thresholds
from src_kaggle.experiments.oof_manager import standardize_oof
from src_kaggle.experiments.experiment_registry import append_experiment
from src_kaggle.experiments.model_comparison import compare_oof_models


def main():
    oof=pd.DataFrame({
        'id':range(8),'term_id':[1,1,2,2,3,3,4,4],'item_id':range(10,18),'label':[1,0,1,0,1,0,1,0],
        'fold':[0,0,1,1,0,0,1,1],'proba':[.9,.2,.7,.4,.6,.3,.8,.1],'model_name':['m']*8,
        'is_short_query':[1,1,0,0,1,1,0,0],'negative_type':['positive','easy','positive','same_brand','positive','attribute_conflict','positive','same_category']
    })
    opt=optimize_oof_thresholds(oof,segment_min_rows=2)
    assert opt.best_threshold > 0 and not opt.curve.empty
    std=standardize_oof(oof,'exp1','m',opt.best_threshold)
    assert 'experiment_name' in std.columns and 'pred_best_threshold' in std.columns
    with tempfile.TemporaryDirectory() as d:
        p1=Path(d)/'oof1.csv'; p2=Path(d)/'oof2.csv'; std.to_csv(p1,index=False)
        std2=std.copy(); std2['proba']=[.8,.1,.6,.45,.7,.2,.75,.2]; std2.to_csv(p2,index=False)
        reg=append_experiment({'experiment_id':'1','experiment_name':'exp1','model_type':'tabular','oof_macro_f1':opt.best_metrics['macro_f1'],'class0_f1':opt.best_metrics['class0_f1'],'class1_f1':opt.best_metrics['class1_f1'],'best_threshold':opt.best_threshold,'oof_path':str(p1)}, Path(d)/'registry.csv')
        assert len(reg)==1
        cmp=compare_oof_models({'exp1':str(p1),'exp2':str(p2)}, Path(d)/'cmp')
        assert 'models' in cmp
    print('experiment engine examples ok')
if __name__=='__main__': main()
