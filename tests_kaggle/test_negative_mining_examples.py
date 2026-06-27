"""Executable examples for multi-layer negative mining.

Run manually:
  python tests_kaggle/test_negative_mining_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.data.attribute_parser import add_attribute_features
from src_kaggle.data.negative_mining import NegativeMiningConfig, mine_negatives
from src_kaggle.data.pair_builder import build_full_item_text


def sample_items() -> pd.DataFrame:
    return pd.DataFrame({
        "item_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "title": [
            "Nike beyaz kadın sneaker", "Nike erkek şort", "Adidas beyaz spor çorap",
            "Erkek suni deri ceket", "Kadın siyah hakiki deri ceket", "Bebek pamuk zıbın",
            "Unisex okul çantası", "Koşu ayakkabısı 42", "Siyah oversize sweatshirt", "Kırmızı elbise",
        ],
        "category": ["sneaker", "şort", "çorap", "ceket", "ceket", "zıbın", "okul çantası", "ayakkabı", "sweatshirt", "elbise"],
        "brand": ["Nike", "Nike", "Adidas", "Derimod", "Derimod", "MiniCo", "BagCo", "RunCo", "SweatCo", "DressCo"],
        "gender": ["Kadın", "Erkek", "Kadın", "Erkek", "Kadın", "Bebek", "Unisex", "Erkek", "Unisex", "Kadın"],
        "age_group": ["Yetişkin", "Yetişkin", "Yetişkin", "Yetişkin", "Yetişkin", "Bebek", "Çocuk", "Yetişkin", "Yetişkin", "Yetişkin"],
        "attributes": [
            "renk: beyaz, stil: spor", "renk: siyah", "renk: beyaz", "renk: siyah, materyal: suni deri",
            "renk: siyah, materyal: hakiki deri", "materyal: pamuk", "renk: siyah", "numara: 42",
            "renk: siyah, stil: oversize", "renk: kırmızı",
        ],
    })


def sample_positives() -> pd.DataFrame:
    pos = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "term_id": [11, 12, 13, 14, 15],
        "item_id": [1, 5, 6, 7, 8],
        "label": [1, 1, 1, 1, 1],
        "query": ["nike beyaz kadın sneaker", "erkek deri ceket", "bebek zıbın", "unisex okul çantası", "42 numara koşu ayakkabısı"],
        "title": ["Nike beyaz kadın sneaker", "Kadın siyah hakiki deri ceket", "Bebek pamuk zıbın", "Unisex okul çantası", "Koşu ayakkabısı 42"],
        "category": ["sneaker", "ceket", "zıbın", "okul çantası", "ayakkabı"],
        "brand": ["Nike", "Derimod", "MiniCo", "BagCo", "RunCo"],
        "gender": ["Kadın", "Kadın", "Bebek", "Unisex", "Erkek"],
        "age_group": ["Yetişkin", "Yetişkin", "Bebek", "Çocuk", "Yetişkin"],
        "attributes": ["renk: beyaz, stil: spor", "renk: siyah, materyal: hakiki deri", "materyal: pamuk", "renk: siyah", "numara: 42"],
    })
    return build_full_item_text(add_attribute_features(pos))


def main() -> None:
    cfg = NegativeMiningConfig(
        seed=7,
        easy_negatives_per_positive=1,
        same_category_negatives_per_positive=1,
        same_brand_negatives_per_positive=1,
        lexical_negatives_per_positive=1,
        attribute_conflict_negatives_per_positive=1,
    )
    negatives, uncertain = mine_negatives(sample_positives(), sample_items(), cfg)
    assert len(negatives) > 0
    assert not any((negatives["term_id"] == 11) & (negatives["item_id"] == 1))
    assert {"easy", "lexical_confusing"}.issubset(set(negatives["negative_type"]))
    assert {"safety_score", "hardness_score", "source_pool", "safety_status"}.issubset(negatives.columns)
    print(negatives[["term_id", "query", "item_id", "title", "negative_type", "safety_status", "hardness_score"]].head(12).to_string(index=False))
    print(f"negative mining examples ok negatives={len(negatives)} uncertain={len(uncertain)}")


if __name__ == "__main__":
    main()
