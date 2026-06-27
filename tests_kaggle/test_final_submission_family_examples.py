from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import json
import pandas as pd

from src_kaggle.final.candidate_selector import select_final_candidates
from src_kaggle.final.ensemble_optimizer import optimize_blends
from src_kaggle.final.submission_family_builder import choose_family_models, materialize_family_artifacts


def test_final_candidate_placeholders_without_oof():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'lb.csv'
        pd.DataFrame([{'experiment_id':'e1','experiment_name':'x','model_family':'tabular','OOF macro-F1':'','class0 F1':'','class1 F1':'','threshold':'','risk_flag':'','strategic_status':'','notes':''}]).to_csv(p, index=False)
        pool = select_final_candidates(p, Path(d)/'missing.csv')
        assert len(pool) == 3
        assert set(pool['candidate_id']) == {'planned_tabular_balanced','planned_class0_defensive','planned_semantic_aggressive'}
        assert pool['selection_status'].eq('not_ready_no_oof').all()


def test_family_metadata_and_weight_normalization():
    pool = pd.DataFrame([
        {'candidate_id':'m1','experiment_id':'e1','experiment_name':'m1','model_family':'tabular','OOF macro-F1':0.7,'class0 F1':0.66,'class1 F1':0.74,'threshold':0.47,'seed_std':0.001,'risk_label':'','selection_score':0.8,'selection_status':'ready','oof_path':'','test_pred_path':''},
        {'candidate_id':'m2','experiment_id':'e2','experiment_name':'m2','model_family':'dense_enhanced','OOF macro-F1':0.71,'class0 F1':0.64,'class1 F1':0.78,'threshold':0.43,'seed_std':0.002,'risk_label':'','selection_score':0.7,'selection_status':'ready','oof_path':'','test_pred_path':''},
    ])
    blends, report = optimize_blends(pool)
    fam = choose_family_models(pool, blends)
    for f in fam.values():
        w = f.get('weights', {})
        assert abs(sum(w.values()) - 1.0) < 1e-9
    with tempfile.TemporaryDirectory() as d:
        df = materialize_family_artifacts(fam, pool, submission_pairs_path=None, artifact_root=Path(d)/'art', configs_root=Path(d)/'cfg', registry_path=Path(d)/'registry.csv')
        assert len(df) == 3
        for path in df['artifact_dir']:
            meta = json.loads(Path(path, 'metadata.json').read_text())
            assert 'family_name' in meta and 'blend_weights' in meta and 'status' in meta

if __name__ == '__main__':
    test_final_candidate_placeholders_without_oof(); test_family_metadata_and_weight_normalization(); print('final submission family examples ok')
