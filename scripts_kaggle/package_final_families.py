#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_kaggle.finalization.config_freezer import freeze_family_configs
from src_kaggle.finalization.family_packager import package_final_families
from src_kaggle.finalization.artifact_manifest import build_final_artifact_manifest
from src_kaggle.finalization.submission_day_ops import write_operational_decision_table

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config-dir', default='configs/kaggle/final'); p.add_argument('--source-root', default='artifacts/final_submissions'); p.add_argument('--final-root', default='artifacts/final'); p.add_argument('--reports-dir', default='reports/final'); args=p.parse_args()
    freeze_family_configs(args.config_dir, args.source_root, args.reports_dir)
    records=package_final_families(args.source_root, args.final_root)
    manifest=build_final_artifact_manifest(args.config_dir, args.final_root, args.reports_dir)
    write_operational_decision_table(Path(args.reports_dir)/'final_operational_decision_table.md')
    print(f"[OK] packaged families={len(records)} manifest_status={manifest.get('final_release_status')}")
