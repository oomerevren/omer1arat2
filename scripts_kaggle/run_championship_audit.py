#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.audit.championship_audit import write_reports

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out-dir', default='reports/final'); args=p.parse_args()
    res=write_reports(args.out_dir)
    dec=res['decision']
    print(f"[OK] championship audit decision={dec['final_decision']} components={len(res['component_table'])} gaps={len(res['gap_list'])}")
