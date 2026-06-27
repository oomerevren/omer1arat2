from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import pandas as pd

from src_kaggle.ablation.ablation_specs import all_ablation_specs
from src_kaggle.ablation.component_toggle import apply_toggles
from src_kaggle.ablation.ablation_reporting import finalize_master, write_ablation_reports


def test_ablation_specs_and_toggles():
    specs = all_ablation_specs()
    assert len(specs) >= 50
    ids = [s.ablation_id for s in specs]
    assert len(ids) == len(set(ids))
    base = {
        'feature_engineering': {'use_lexical_features': True, 'use_retrieval_features': True, 'use_semantic_features': True},
        'retrieval': {'enabled_retrievers': ['bm25','dense'], 'dense': {'enabled': True}},
        'negative_mining': {},
        'validation_framework': {},
    }
    cfg = apply_toggles(base, {'feature_disable': ['use_retrieval_features'], 'retrieval_mode': 'bm25_only'})
    assert cfg['feature_engineering']['use_retrieval_features'] is True  # retrieval_mode re-enables BM25 retrieval features
    assert cfg['retrieval']['enabled_retrievers'] == ['bm25']
    assert cfg['retrieval']['dense']['enabled'] is False


def test_ablation_reporting_outputs():
    rows = [
        {'ablation_id':'feat_all_features','experiment_name':'tab_all','category':'feature','changed_component':'all','variant_description':'all','splitter':'term_group','seed_set':'42','OOF macro-F1':0.70,'class0 F1':0.65,'class1 F1':0.75,'best_threshold':0.47,'threshold_fragility':False,'seed_std':0.001,'public_lb_score':'','public_oof_delta':'','note':'','status':'completed','report_dir':'','oof_path':''},
        {'ablation_id':'feat_no_retrieval','experiment_name':'tab_no_ret','category':'feature','changed_component':'retrieval','variant_description':'no retrieval','splitter':'term_group','seed_set':'42','OOF macro-F1':0.69,'class0 F1':0.63,'class1 F1':0.75,'best_threshold':0.50,'threshold_fragility':False,'seed_std':0.001,'public_lb_score':'','public_oof_delta':'','note':'','status':'completed','report_dir':'','oof_path':''},
    ]
    master = finalize_master(rows)
    assert 'risk_flag' in master.columns
    with tempfile.TemporaryDirectory() as d:
        write_ablation_reports(master, d)
        assert Path(d, 'ablation_master_table.csv').exists()
        assert Path(d, 'final_pipeline_recommendation.md').exists()

if __name__ == '__main__':
    test_ablation_specs_and_toggles(); test_ablation_reporting_outputs(); print('ablation framework examples ok')
