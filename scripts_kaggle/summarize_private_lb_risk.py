#!/usr/bin/env python
"""Rebuild private-LB strategy summaries from current registries."""
from __future__ import annotations
import subprocess, sys

if __name__ == '__main__':
    subprocess.check_call([sys.executable, 'scripts_kaggle/analyze_leaderboard_correlation.py'])
