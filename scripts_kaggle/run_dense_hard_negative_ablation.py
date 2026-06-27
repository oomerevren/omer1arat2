#!/usr/bin/env python
"""Small dense hard-negative ablation launcher placeholder."""
from __future__ import annotations
import argparse, subprocess, sys

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/kaggle/war_mode.yaml'); args=p.parse_args()
    subprocess.check_call([sys.executable, 'scripts_kaggle/build_negatives.py', '--config', args.config, '--use-dense', 'true'])
    print('[OK] dense hard negative ablation seed run completed; compare OOF in next sprint.')
if __name__=='__main__': main()
