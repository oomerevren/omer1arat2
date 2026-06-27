#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.finalization.final_release_validator import validate_final_release

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config-dir', default='configs/kaggle/final'); p.add_argument('--final-root', default='artifacts/final'); p.add_argument('--reports-dir', default='reports/final'); p.add_argument('--strict', action='store_true'); args=p.parse_args()
    report=validate_final_release(args.config_dir, args.final_root, args.reports_dir)
    print(f"[OK] metadata_lock_ready={report['metadata_lock_ready']} release_ready={report['release_ready']} errors={len(report['errors'])} warnings={len(report['warnings'])}")
    if args.strict and not report['release_ready']:
        raise SystemExit(2)
