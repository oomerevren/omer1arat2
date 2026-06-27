from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import pandas as pd

from src_kaggle.leaderboard.oof_public_linker import append_public_lb_entry, build_oof_public_table
from src_kaggle.leaderboard.risk_flagging import apply_risk_flags
from src_kaggle.leaderboard.splitter_reliability import splitter_reliability
from src_kaggle.leaderboard.model_family_drift import model_family_drift
from src_kaggle.leaderboard.threshold_reliability import threshold_reliability


def test_public_oof_join_and_risk_flags():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        exp = pd.DataFrame([
            {'experiment_id':'e1','experiment_name':'tab_base','model_type':'tabular','validation_version':'term_group','best_threshold':0.47,'oof_macro_f1':0.70,'class0_f1':0.65,'class1_f1':0.75,'threshold_fragile':False,'report_dir':'','oof_path':'','public_lb_score':''},
            {'experiment_id':'e2','experiment_name':'dense_aggressive','model_type':'tabular','validation_version':'term_group','best_threshold':0.91,'oof_macro_f1':0.69,'class0_f1':0.62,'class1_f1':0.76,'threshold_fragile':True,'report_dir':'','oof_path':'','public_lb_score':''},
        ])
        exp_path = d/'experiments.csv'; exp.to_csv(exp_path, index=False)
        sub_path = d/'subs.csv'; pd.DataFrame().to_csv(sub_path, index=False)
        abl_path = d/'abl.csv'; pd.DataFrame().to_csv(abl_path, index=False)
        pub_path = d/'public.csv'
        append_public_lb_entry(experiment_id='e1', public_lb_score=0.71, file_path='s1.csv', tracking_path=pub_path)
        append_public_lb_entry(experiment_id='e2', public_lb_score=0.72, file_path='s2.csv', tracking_path=pub_path)
        table = build_oof_public_table(experiment_registry=exp_path, submission_registry=sub_path, ablation_master=abl_path, public_lb_tracking=pub_path)
        out = apply_risk_flags(table)
        assert len(out) == 2
        assert out['public_lb_score'].notna().sum() == 2
        e2 = out[out['experiment_id'].eq('e2')].iloc[0]
        assert 'PUBLIC_UP_OOF_DOWN' in e2['risk_flag']
        assert 'PUBLIC_UP_CLASS0_DOWN' in e2['risk_flag']
        assert 'THRESHOLD_FRAGILE' in e2['risk_flag']


def test_reliability_reports_smoke():
    df = pd.DataFrame([
        {'experiment_id':'e1','experiment_name':'tab','model_family':'tabular','splitter':'term_group','threshold':0.5,'OOF macro-F1':0.70,'class0 F1':0.65,'class1 F1':0.75,'public_lb_score':0.71,'public_minus_oof':0.01,'threshold_fragility':False,'seed_std':0.001,'risk_flag':''},
        {'experiment_id':'e2','experiment_name':'dense','model_family':'dense_enhanced','splitter':'term_group','threshold':0.9,'OOF macro-F1':0.69,'class0 F1':0.62,'class1 F1':0.76,'public_lb_score':0.72,'public_minus_oof':0.03,'threshold_fragility':True,'seed_std':0.002,'risk_flag':'THRESHOLD_FRAGILE|DENSE_ARTIFACT_RISK'},
    ])
    sp, spr = splitter_reliability(df)
    fam, famr = model_family_drift(df)
    thr, thrr = threshold_reliability(df)
    assert not sp.empty and spr['recommended_splitter'] == 'term_group'
    assert set(fam['model_family']) == {'tabular','dense_enhanced'}
    assert 'threshold_extreme_flag' in thr.columns

if __name__ == '__main__':
    test_public_oof_join_and_risk_flags(); test_reliability_reports_smoke(); print('leaderboard intelligence examples ok')
