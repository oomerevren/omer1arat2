import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment.config_loader import load_config
from src.training.trainer import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Teknofest 2026 deney çalıştırıcı")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--mode", type=str, choices=["kaggle", "final"], help="Yarışma modu override")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Config: {args.config}")
    config = load_config(args.config)

    if args.mode:
        config.setdefault("experiment", {})["mode"] = args.mode

    exp_name = config.get("experiment", {}).get("name", "baseline")
    print(f"Deney başlatılıyor: {exp_name} (mod: {config.get('experiment', {}).get('mode', 'final')})")

    result = run_experiment(config)
    print(f"\nDeney tamamlandı.")
    print(f"  Run ID: {result['run_id']}")
    print(f"  Metrikler: {result['metrics']}")
    print(f"  Model: {result['save_path']}")


if __name__ == "__main__":
    main()
