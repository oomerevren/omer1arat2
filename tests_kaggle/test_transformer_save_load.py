from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src_kaggle.models.transformer_checkpointing import write_training_config, write_metrics
import tempfile, json

def main():
    with tempfile.TemporaryDirectory() as d:
        c=write_training_config(Path(d)/'training_config.json',{'model_name':'x','fold':0})
        m=write_metrics(Path(d)/'metrics.json',{'macro_f1':0.5})
        assert json.loads(Path(c).read_text())['model_name']=='x'
        assert json.loads(Path(m).read_text())['macro_f1']==0.5
    print('transformer checkpoint metadata ok')
if __name__=='__main__': main()
