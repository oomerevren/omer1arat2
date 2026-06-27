"""
Kaggle submission üretici — 26 Haziran'da gerçek test.csv gelince çalıştırılır.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_csv
from src.experiment.config_loader import load_config
from src.models.cross_encoder import CrossEncoderModel
from src.training.trainer import preprocess_dataframe, resolve_model_path
from src.utils.io import read_json


def parse_args():
    p = argparse.ArgumentParser(description="Kaggle submission CSV üret")
    p.add_argument("--config", default="configs/model/kaggle.yaml")
    p.add_argument("--test", default="./data/test.csv")
    p.add_argument("--model", default=None, help="Model dizini (varsayılan: config experiment adından)")
    p.add_argument("--output", default="./submissions/submission.csv")
    p.add_argument("--threshold", type=float, default=None, help="Binary eşik (yoksa metrics.json'dan)")
    return p.parse_args()


def _load_threshold(model_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    metrics_path = model_dir.parent / "metrics.json"
    if metrics_path.exists():
        data = read_json(metrics_path)
        return float(data.get("threshold_class_1", data.get("threshold_positive", 0.5)))
    return 0.5


def main():
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("experiment", {})["mode"] = "kaggle"

    if not os.path.exists(args.test):
        print(f"[HATA] Test dosyası bulunamadı: {args.test}")
        print("Organizatör verisi 26 Haziran 2026'da iletilecek.")
        sys.exit(1)

    model_path = Path(args.model) if args.model else resolve_model_path(config)
    test_df = load_csv(args.test, config)
    test_df = preprocess_dataframe(test_df, config)

    if model_path.exists():
        model = CrossEncoderModel.load(model_path)
    else:
        print(f"[UYARI] Eğitilmiş model yok ({model_path}), pretrained kullanılıyor.")
        ce_cfg = config.get("model", {}).get("cross_encoder", {})
        model = CrossEncoderModel(
            model_name=ce_cfg.get("model_name", "dbmdz/distilbert-base-turkish-cased"),
            num_labels=2,
        )

    threshold = _load_threshold(model_path, args.threshold)
    probs = model.predict_proba(test_df, config)
    preds = (probs[:, 1] >= threshold).astype(int) if probs.shape[1] == 2 else (probs.argmax(axis=1) >= 1).astype(int)

    id_col = config.get("data", {}).get("id_column", "product_id")
    q_col = config.get("data", {}).get("query_column", "search_query")
    submit_cols = [c for c in [id_col, q_col, "prediction"] if c in test_df.columns or c == "prediction"]
    out = test_df.copy()
    out["prediction"] = preds

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out[submit_cols].to_csv(args.output, index=False)
    print(f"[OK] Submission: {args.output} ({len(out)} satır, threshold={threshold:.3f})")


if __name__ == "__main__":
    main()
