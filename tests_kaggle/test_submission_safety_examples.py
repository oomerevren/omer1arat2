from __future__ import annotations
import sys, tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_kaggle.submission.submission_builder import build_submission_from_proba, save_safe_submission, append_submission_registry
from src_kaggle.submission.submission_validator import validate_submission

def main():
    ref=pd.DataFrame({'id':[1,2,3]}); sub=build_submission_from_proba(ref,[.2,.8,.6],.5)
    res=validate_submission(sub,ref); assert res.is_valid and list(sub.columns)==['id','prediction']
    bad=pd.DataFrame({'id':[1,2,2],'prediction':[1,0,1]}); assert not validate_submission(bad,ref).is_valid
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)/'sub.csv'; rep=Path(d)/'rep.json'; save_safe_submission(sub,ref,out,report_path=rep); assert out.exists() and rep.exists()
        reg=append_submission_registry({'file_path':str(out),'positive_rate':res.positive_rate},Path(d)/'registry.csv'); assert len(reg)==1
    print('submission safety examples ok')
if __name__=='__main__': main()
