"""Executable examples for src_kaggle.data.attribute_parser.

Run manually:
  python tests_kaggle/test_attribute_parser_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_kaggle.data.attribute_parser import parse_attributes


def check(raw, expected_subset):
    parsed = parse_attributes(raw)
    data = parsed.attribute_dict
    for key, expected_values in expected_subset.items():
        assert data.get(key) == expected_values, (raw, data, expected_subset)
    return parsed


def main() -> None:
    # Empty / null tolerant
    assert parse_attributes("").attribute_dict == {}
    assert parse_attributes(None).attribute_dict == {}

    # Key aliases + value normalization
    p = check("renk: siyah, materyal: hakiki deri, stil: klasik", {
        "color": ["black"],
        "material": ["genuine_leather"],
        "style": ["classic"],
    })
    assert p.color_value == "black"
    assert p.material_value == "genuine_leather"
    assert "color: black" in p.normalized_attribute_text

    # English aliases + case/spacing
    check("COLOR: White, Fabric: Cotton", {"color": ["white"], "material": ["cotton"]})

    # Turkish character / ASCII tolerant key matching
    check("Ürün Rengi: Kırmızı, Kumaş Tipi: Pamuk", {"color": ["red"], "material": ["cotton"]})

    # Repeated key + multi value dedupe
    check("renk: siyah / beyaz, color: black", {"color": ["black", "white"]})

    # Semantic material differences are preserved
    check("materyal: deri, materyal: hakiki deri, materyal: suni deri", {
        "material": ["leather", "genuine_leather", "faux_leather"]
    })

    # Malformed fragments are preserved under unknown
    malformed = parse_attributes("renk siyah, materyal: deri")
    assert malformed.attribute_dict["unknown"] == ["renk_siyah"]
    assert malformed.attribute_dict["material"] == ["leather"]

    print("attribute parser examples ok")


if __name__ == "__main__":
    main()
