"""
Kaggle verisi gelene kadar geliştirme/test için örnek CSV üretir.
26 Haziran'da gerçek train.csv geldiğinde bu dosya kullanılmaz.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _sample_rows() -> pd.DataFrame:
    queries = [
        "siyah nike spor ayakkabı",
        "kırmızı elbise kadın",
        "mavi kot pantolon erkek",
        "beyaz tişört oversize",
        "deri çanta kahverengi",
        "adidas ayakkabı beyaz",
        "siyah deri ceket erkek",
        "kışlık mont columbia",
        "topuklu ayakkabı siyah",
        "puma spor ayakkabı",
        "iphone 15 kılıf",
        "samsung telefon kılıfı",
        "çocuk mont kırmızı",
        "erkek gömlek mavi",
        "kadın bot siyah",
    ]
    products = [
        ("Nike Air Max Siyah Spor Ayakkabı", "Nike", "Ayakkabı > Spor", "Siyah", "Deri"),
        ("Siyah Uzun Elbise Kadın Pamuk", "TrendyolMilla", "Giyim > Elbise", "Siyah", "Pamuk"),
        ("Levi's Mavi Kot Pantolon", "Levi's", "Giyim > Pantolon", "Mavi", "Denim"),
        ("Beyaz Basic Tişört", "TrendyolMilla", "Giyim > Tişört", "Beyaz", "Pamuk"),
        ("Kırmızı Deri Çanta", "Derimod", "Aksesuar > Çanta", "Kırmızı", "Deri"),
        ("Adidas Ultraboost Beyaz", "Adidas", "Ayakkabı > Spor", "Beyaz", "Mesh"),
        ("Deri Ceket Erkek Siyah", "Derimod", "Giyim > Ceket", "Siyah", "Deri"),
        ("Kışlık Su Geçirmez Mont", "Columbia", "Giyim > Mont", "Lacivert", "Polyester"),
        ("Topuklu Ayakkabı Siyah", "İnci", "Ayakkabı > Topuklu", "Siyah", "Deri"),
        ("Spor Ayakkabı Puma", "Puma", "Ayakkabı > Spor", "Beyaz", "Mesh"),
        ("iPhone 15 Silikon Kılıf", "Apple", "Elektronik > Aksesuar", "Şeffaf", "Silikon"),
        ("Samsung Galaxy Kılıf", "Samsung", "Elektronik > Aksesuar", "Siyah", "Plastik"),
        ("Çocuk Kışlık Mont", "Koton", "Giyim > Mont", "Kırmızı", "Polyester"),
        ("Erkek Slim Fit Gömlek", "D'S Damat", "Giyim > Gömlek", "Mavi", "Pamuk"),
        ("Kadın Bot Siyah Deri", "Hotiç", "Ayakkabı > Bot", "Siyah", "Deri"),
    ]

    rows = []
    for i, (q, (name, brand, cat, color, mat)) in enumerate(zip(queries, products)):
        rows.append({
            "product_id": f"p_{i:04d}",
            "search_query": q,
            "product_name": name,
            "brand": brand,
            "category": cat,
            "product_color": color,
            "product_material": mat,
            "is_relevant": 2 if i < 10 else (1 if i < 13 else 0),
        })

    # Negatif örnekler ekle
    neg_rows = [
        {"product_id": "p_neg_01", "search_query": "nike spor ayakkabı", "product_name": "Adidas Ayakkabı",
         "brand": "Adidas", "category": "Ayakkabı > Spor", "product_color": "Siyah", "product_material": "Mesh", "is_relevant": 0},
        {"product_id": "p_neg_02", "search_query": "deri çanta kadın", "product_name": "Deri Ayakkabı",
         "brand": "Derimod", "category": "Ayakkabı > Günlük", "product_color": "Kahverengi", "product_material": "Deri", "is_relevant": 0},
        {"product_id": "p_neg_03", "search_query": "kırmızı elbise", "product_name": "Mavi Kot Pantolon",
         "brand": "Levi's", "category": "Giyim > Pantolon", "product_color": "Mavi", "product_material": "Denim", "is_relevant": 0},
    ]
    rows.extend(neg_rows)
    return pd.DataFrame(rows)


def _sample_rows_for_config(config: Dict[str, Any]) -> pd.DataFrame:
    df = _sample_rows()
    mode = config.get("experiment", {}).get("mode", "final")
    if mode == "kaggle":
        df["is_relevant"] = (df["is_relevant"] >= 1).astype(int)
    return df


def ensure_sample_data(config: Dict[str, Any]) -> str:
    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path", "./data/train.csv")
    test_path = data_cfg.get("test_path", "./data/test.csv")

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(train_path):
        df = _sample_rows_for_config(config)
        df.to_csv(train_path, index=False, encoding="utf-8")
        print(f"[SampleData] Örnek eğitim verisi: {train_path} ({len(df)} satır)")
    elif config.get("experiment", {}).get("mode") == "kaggle":
        existing = pd.read_csv(train_path)
        if "is_relevant" in existing.columns and existing["is_relevant"].max() > 1:
            existing["is_relevant"] = (existing["is_relevant"] >= 1).astype(int)
            existing.to_csv(train_path, index=False, encoding="utf-8")
            print(f"[SampleData] Kaggle modu için etiketler binary'e dönüştürüldü: {train_path}")

    if not os.path.exists(test_path):
        test_df = _sample_rows_for_config(config).head(8).copy()
        if config.get("experiment", {}).get("mode") == "kaggle":
            test_df["is_relevant"] = (test_df["is_relevant"] >= 1).astype(int)
        else:
            test_df["is_relevant"] = [2, 2, 0, 1, 2, 0, 1, 0]
        test_df.to_csv(test_path, index=False, encoding="utf-8")
        print(f"[SampleData] Örnek test verisi: {test_path}")

    return train_path
