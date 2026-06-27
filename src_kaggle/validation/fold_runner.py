from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from src_kaggle.models.tabular_model import TabularModel
from src_kaggle.models.cross_encoder_model import CrossEncoderModel
from src_kaggle.validation.splitters import get_splitter
from src_kaggle.validation.threshold_tuning import find_best_threshold, metrics_at_threshold

class ValidationFoldRunner:
    def __init__(self, model_type='tabular', model_cfg=None, feature_cfg=None, items=None, n_splits=5, seed=42, splitter='term_group'):
        self.model_type=model_type; self.model_cfg=model_cfg or {}; self.feature_cfg=feature_cfg or {}; self.items=items; self.n_splits=n_splits; self.seed=seed; self.splitter=splitter

    def _fit_predict(self,tr,va,fold):
        if self.model_type=='cross_encoder':
            m=CrossEncoderModel(backend=self.model_cfg.get('backend','sklearn_text'), model_name=self.model_cfg.get('model_name','dbmdz/distilbert-base-turkish-cased'), text_format_version=self.model_cfg.get('text_format_version','full_v1'), seed=self.seed+fold, params=self.model_cfg.get('params',{}))
            m.fit(tr,tr[SCHEMA.label].astype(int)); return m.predict_proba(va), m
        fp=FeaturePipeline(FeaturePipelineConfig.from_dict(self.feature_cfg), items=self.items)
        Xtr=fp.transform(tr).features; Xva=fp.transform(va).features
        m=TabularModel(model_type=self.model_cfg.get('model_type','hist_gradient_boosting'), params=self.model_cfg.get('params',{}), seed=self.seed+fold)
        m.fit(Xtr,tr[SCHEMA.label].astype(int)); return m.predict_proba(Xva), m

    def run(self,df:pd.DataFrame):
        splitter=get_splitter(self.splitter,self.n_splits,self.seed)
        oof=np.zeros(len(df)); folds=np.full(len(df),-1); fold_rows=[]
        for fold,(tr_idx,va_idx) in enumerate(splitter.split(df)):
            tr=df.iloc[tr_idx].reset_index(drop=True); va=df.iloc[va_idx].reset_index(drop=True)
            pred,_=self._fit_predict(tr,va,fold)
            oof[va_idx]=pred; folds[va_idx]=fold
            th,m,_=find_best_threshold(va[SCHEMA.label], pred, step=0.01)
            m.update({'fold':fold,'n_train':len(tr),'n_val':len(va),'val_positive_rate':float(va[SCHEMA.label].mean()),'fold_best_threshold':th})
            fold_rows.append(m)
        best_th,best_metrics,curve=find_best_threshold(df[SCHEMA.label],oof,step=0.01)
        return {'proba':oof,'folds':folds,'best_threshold':best_th,'overall_metrics':best_metrics,'threshold_curve':curve,'fold_scores':pd.DataFrame(fold_rows)}
