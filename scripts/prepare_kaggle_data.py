#!/usr/bin/env python
"""
26 Haziran 2026 sonrası — organizatör Kaggle verisini pipeline'a bağlar.

Kullanım:
  python scripts/prepare_kaggle_data.py --train /path/to/train.csv --test /path/to/test.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_csv, normalize_dataframe
from src.experiment.config_loader import load_config


def main():
    p = argparse.ArgumentParser(description="Kaggle verisini data/ klasörüne hazırla")
    p.add_argument("--train", required=True, help="Organizatör train.csv")
    p.add_argument("--test", required=True, help="Organizatör test.csv")
    p.add_argument("--config", default="configs/model/kaggle.yaml")
    args = p.parse_args()

    config = load_config(args.config)
    config.setdefault("experiment", {})["mode"] = "kaggle"

    train_df = normalize_dataframe(load_csv(args.train, config), config)
    test_df = normalize_dataframe(load_csv(args.test, config), config)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    train_out = data_dir / "train.csv"
    test_out = data_dir / "test.csv"

    train_df.to_csv(train_out, index=False, encoding="utf-8")
    test_df.to_csv(test_out, index=False, encoding="utf-8")

    print(f"[OK] Eğitim: {train_out} ({len(train_df)} satır)")
    print(f"[OK] Test: {test_out} ({len(test_df)} satır)")
    print("\nSonraki adımlar:")
    print("  python scripts/run_experiment.py --config configs/model/kaggle.yaml")
    print("  python scripts/kaggle_submission.py --config configs/model/kaggle.yaml")


if __name__ == "__main__":
    main()
