"""
Esnek veri seti yükleyici — Kaggle verisi gelene kadar örnek veri destekler.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_COLUMNS = {
    "query_column": "search_query",
    "product_text_column": "product_name",
    "label_column": "is_relevant",
    "id_column": "product_id",
    "category_column": "category",
    "brand_column": "brand",
    "color_column": "product_color",
    "material_column": "product_material",
}

# Olası alternatif sütun adları (organizatör formatına uyum)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "search_query": ["search_query", "query", "search_term", "term", "arama_terimi"],
    "product_name": ["product_name", "title", "product_title", "urun_adi", "name"],
    "is_relevant": ["is_relevant", "label", "relevance", "alaka", "target"],
    "product_id": ["product_id", "id", "urun_id", "item_id"],
    "category": ["category", "category_name", "kategori", "category_hierarchy"],
    "brand": ["brand", "marka", "brand_name"],
    "product_color": ["product_color", "color", "renk"],
    "product_material": ["product_material", "material", "materyal"],
}


def _resolve_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    aliases = COLUMN_ALIASES.get(canonical, [canonical])
    for name in aliases:
        if name in df.columns:
            return name
    return None


def normalize_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Organizatör CSV'sini standart sütun adlarına dönüştürür."""
    data_cfg = {**DEFAULT_COLUMNS, **config.get("data", {})}
    out = df.copy()

    mapping = {}
    for canonical in COLUMN_ALIASES:
        found = _resolve_column(out, canonical)
        if found and found != canonical:
            mapping[found] = canonical

    if mapping:
        out = out.rename(columns=mapping)

    # Eksik sütunları boş ekle
    for col in DEFAULT_COLUMNS.values():
        if col not in out.columns:
            out[col] = "" if col != "is_relevant" else 0

    label_col = data_cfg["label_column"]
    if label_col in out.columns:
        out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)

    if config:
        from src.data.labels import normalize_labels
        out = normalize_labels(out, config)

    return out


def build_product_text(row: pd.Series, config: Dict[str, Any]) -> str:
    """Sorgu-ürün çifti için birleşik ürün metni oluşturur."""
    data_cfg = {**DEFAULT_COLUMNS, **config.get("data", {})}
    parts = [
        str(row.get(data_cfg["product_text_column"], "")),
        str(row.get(data_cfg["category_column"], "")),
        str(row.get(data_cfg["brand_column"], "")),
        str(row.get(data_cfg["color_column"], "")),
        str(row.get(data_cfg["material_column"], "")),
    ]
    return " | ".join(p.strip() for p in parts if p and p.strip() and p != "nan")


def load_csv(path: str, config: Dict[str, Any]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    return normalize_dataframe(df, config)


def load_train_test(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path", "./data/train.csv")
    test_path = data_cfg.get("test_path", "./data/test.csv")

    if not os.path.exists(train_path):
        from src.data.sample_generator import ensure_sample_data

        ensure_sample_data(config)

    train_df = load_csv(train_path, config)
    test_df = load_csv(test_path, config) if os.path.exists(test_path) else None
    return train_df, test_df


def get_num_labels(config: Dict[str, Any]) -> int:
    mode = config.get("experiment", {}).get("mode", "final")
    if mode == "kaggle":
        return 2
    return config.get("model", {}).get("cross_encoder", {}).get("num_labels", 3)
