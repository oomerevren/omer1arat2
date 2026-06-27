"""Executable examples for query intent extraction.

Run manually:
  python tests_kaggle/test_query_intent_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.features.query_intent import build_query_intent_resources, extract_query_intent


ITEMS = pd.DataFrame(
    {
        "item_id": [1, 2, 3, 4],
        "title": ["Nike sneaker", "Bebek zıbın", "Deri ceket", "Okul çantası"],
        "category": ["sneaker", "zıbın", "ceket", "okul çantası"],
        "brand": ["Nike", "MiniCo", "Derimod", "BagCo"],
        "gender": ["Kadın", "Bebek", "Erkek", "Unisex"],
        "age_group": ["Yetişkin", "Bebek", "Yetişkin", "Çocuk"],
        "attributes": [
            "renk: beyaz, stil: spor",
            "materyal: pamuk",
            "materyal: deri",
            "renk: siyah",
        ],
    }
)


def main() -> None:
    resources = build_query_intent_resources(ITEMS)

    q1 = extract_query_intent("nike beyaz kadın sneaker", resources)
    assert q1["has_brand_token"] == 1
    assert q1["has_color_token"] == 1
    assert q1["has_gender_token"] == 1
    assert q1["has_category_token"] == 1

    q2 = extract_query_intent("bebek zıbın", resources)
    assert q2["has_age_token"] == 1
    assert q2["has_category_token"] == 1
    assert q2["is_short_query"] == 1

    q3 = extract_query_intent("erkek deri ceket", resources)
    assert q3["has_gender_token"] == 1
    assert q3["has_material_token"] == 1
    assert q3["has_category_token"] == 1

    q4 = extract_query_intent("unisex okul çantası", resources)
    assert q4["has_gender_token"] == 1
    assert q4["has_age_token"] == 1
    assert q4["has_category_token"] == 1

    q5 = extract_query_intent("42 numara koşu ayakkabısı", resources)
    assert q5["has_size_token"] == 1
    assert q5["has_category_token"] == 1

    q6 = extract_query_intent("siyah oversize sweatshirt", resources)
    assert q6["has_color_token"] == 1
    assert q6["has_style_token"] == 1
    assert q6["has_category_token"] == 1
    assert q6["is_attribute_heavy"] == 1

    print("query intent examples ok")


if __name__ == "__main__":
    main()
