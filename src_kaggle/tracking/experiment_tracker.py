"""Real CSV/Markdown experiment tracking utilities."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def _md_table(df):
    if df.empty: return '_no rows_'
    cols=list(df.columns)
    lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for _,r in df.iterrows(): lines.append('|'+'|'.join(str(r.get(c,'')) for c in cols)+'|')
    return '\n'.join(lines)

TRACKING_COLUMNS=[
 'run_id','timestamp','git_commit','experiment_name','model_type','data_version','negative_mining_version','retrieval_version','feature_version','validation_version','seed','oof_macro_f1','class0_f1','class1_f1','threshold','pseudo_labeling_mode','pseudo_label_count','config_path','oof_path','error_report','pseudo_report','artifact_dir','notes'
]

def git_commit()->str:
    try: return subprocess.check_output(['git','rev-parse','--short','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: return ''

def log_run(record:dict,path='reports/experiments/master_experiment_log.csv')->pd.DataFrame:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    row={c:record.get(c,'') for c in TRACKING_COLUMNS}; row['timestamp']=row.get('timestamp') or datetime.now(timezone.utc).isoformat(); row['git_commit']=row.get('git_commit') or git_commit()
    old=pd.read_csv(p) if p.exists() else pd.DataFrame(columns=TRACKING_COLUMNS)
    df=pd.concat([old,pd.DataFrame([row])],ignore_index=True); df.to_csv(p,index=False); write_master_markdown(df,p.with_suffix('.md')); return df

def write_master_markdown(df:pd.DataFrame,path:Path):
    top=df.copy()
    if 'oof_macro_f1' in top: top['oof_macro_f1']=pd.to_numeric(top['oof_macro_f1'],errors='coerce')
    top=top.sort_values('oof_macro_f1',ascending=False,na_position='last').head(30)
    lines=['# Master Experiment Log','','## Top runs','',_md_table(top)]
    pseudo=df[df.get('pseudo_label_count','').astype(str).ne('')] if not df.empty and 'pseudo_label_count' in df else pd.DataFrame()
    if not pseudo.empty: lines += ['','## Runs with pseudo-labeling','',_md_table(pseudo.tail(20))]
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def log_from_experiment_registry_row(row:dict, extra:dict|None=None):
    extra=extra or {}; rec={
        'run_id': row.get('experiment_id',''), 'experiment_name': row.get('experiment_name',''), 'model_type': row.get('model_type',''),
        'data_version': row.get('data_version',''), 'negative_mining_version': row.get('negative_mining_version',''), 'retrieval_version': row.get('retrieval_version',''), 'feature_version': row.get('feature_version',''), 'validation_version': row.get('validation_version',''),
        'seed': row.get('seed',''), 'oof_macro_f1': row.get('oof_macro_f1',''), 'class0_f1': row.get('class0_f1',''), 'class1_f1': row.get('class1_f1',''), 'threshold': row.get('best_threshold',''), 'oof_path': row.get('oof_path',''), 'artifact_dir': row.get('report_dir','')
    }; rec.update(extra); return log_run(rec)
