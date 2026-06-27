#!/usr/bin/env python
"""Print final submission family summary."""
from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--families', default='reports/final/final_submission_families.csv'); args=p.parse_args()
    path=Path(args.families)
    if not path.exists():
        raise SystemExit(f'Missing {path}; run build_final_submission_families.py first')
    df=pd.read_csv(path)
    print(df[['family_name','public_private_risk_label','status','used_models','threshold']].to_string(index=False))
if __name__=='__main__': main()
