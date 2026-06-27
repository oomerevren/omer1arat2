"""Controlled pseudo-labeling with risk reports."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src_kaggle.pseudo_labeling.confidence_filters import pseudo_label_decision
from src_kaggle.data.io import write_table
from src_kaggle.data.schema import SCHEMA


def generate_pseudo_labels(candidates: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    if cfg.get('mode','disabled')=='disabled':
        return pd.DataFrame(), {'mode':'disabled','pseudo_count':0}
    model_cols=cfg.get('model_probability_columns', [])
    rows=[]; rejected=[]
    for _,r in candidates.iterrows():
        label,reason,agree=pseudo_label_decision(r,cfg,model_cols)
        rec=r.to_dict(); rec['pseudo_reason']=reason; rec['agreement_score']=agree; rec['margin']=abs(float(r.get('proba',0.5))-0.5)
        if label is None:
            rejected.append(rec); continue
        rec[SCHEMA.label]=label; rec['pseudo_label']=label; rows.append(rec)
    pseudo=pd.DataFrame(rows); rej=pd.DataFrame(rejected)
    report={
        'mode': cfg.get('mode'), 'pseudo_count': int(len(pseudo)), 'rejected_count': int(len(rej)),
        'positive_count': int((pseudo.get('pseudo_label',pd.Series(dtype=int))==1).sum()) if not pseudo.empty else 0,
        'negative_count': int((pseudo.get('pseudo_label',pd.Series(dtype=int))==0).sum()) if not pseudo.empty else 0,
        'positive_threshold': cfg.get('positive_threshold',0.95), 'negative_threshold': cfg.get('negative_threshold',0.05),
        'segment_distribution': {c: pseudo[c].value_counts(dropna=False).head(20).to_dict() for c in ['is_short_query','is_brand_heavy','is_attribute_heavy','negative_type',SCHEMA.category] if c in pseudo.columns} if not pseudo.empty else {},
        'agreement_summary': pseudo['agreement_score'].describe().to_dict() if 'agreement_score' in pseudo else {},
        'margin_summary': pseudo['margin'].describe().to_dict() if 'margin' in pseudo else {},
        'risk_notes': []
    }
    if report['pseudo_count']>0 and report['positive_count']/max(1,report['pseudo_count'])>0.9: report['risk_notes'].append('Pseudo labels are highly positive-skewed; monitor class balance.')
    if report['pseudo_count']>int(cfg.get('max_pseudo_samples_warning',50000)): report['risk_notes'].append('Large pseudo-label volume; use capped ablation first.')
    return pseudo, report


def write_pseudo_outputs(pseudo: pd.DataFrame, report: dict, output_path='data/processed/pseudo_labels.parquet', report_dir='reports/pseudo_labeling') -> dict:
    write_table(pseudo, output_path)
    out=Path(report_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'pseudo_label_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    lines=['# Pseudo Label Report','',f"Mode: {report.get('mode')}",f"Pseudo count: {report.get('pseudo_count')}",f"Positive: {report.get('positive_count')}",f"Negative: {report.get('negative_count')}",'','## Risk notes']
    for n in report.get('risk_notes',[]): lines.append(f'- {n}')
    (out/'pseudo_label_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'pseudo_path':output_path,'report_json':str(out/'pseudo_label_report.json'),'report_md':str(out/'pseudo_label_report.md')}
