"""Attribute parser and normalizer for TEKNOFEST Kaggle items.

Official `items.csv` provides attributes as a single string, commonly formatted as:
    "anahtar: değer, anahtar: değer, ..."

This module converts that fragile string into stable feature-engineering inputs:
- attribute_dict: canonical key -> list of normalized values
- attribute_keys: pipe-separated canonical keys
- attribute_values: pipe-separated normalized values
- normalized_attribute_text: deterministic key-value text
- color_value / material_value / style_value: high-value optional shortcuts

The parser is tolerant: empty, null and malformed records do not crash the
pipeline. Malformed fragments are preserved under the `unknown` key so that
information is not silently lost.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src_kaggle.data.schema import SCHEMA


# Canonical key aliases. Keep this compact and high-signal; long-tail keys are
# normalized with a generic slug function rather than discarded.
KEY_ALIASES: dict[str, str] = {
    # color
    "renk": "color",
    "urun rengi": "color",
    "ürün rengi": "color",
    "color": "color",
    "colour": "color",
    "rengi": "color",
    # material / fabric
    "materyal": "material",
    "malzeme": "material",
    "kumas": "material",
    "kumaş": "material",
    "kumas tipi": "material",
    "kumaş tipi": "material",
    "fabric": "material",
    "material": "material",
    "ic materyal": "material",
    "iç materyal": "material",
    "dis materyal": "material",
    "dış materyal": "material",
    # style
    "stil": "style",
    "style": "style",
    "tarz": "style",
    "model": "style",
    # other useful e-commerce fields
    "desen": "pattern",
    "pattern": "pattern",
    "beden": "size",
    "size": "size",
    "kalip": "fit",
    "kalıp": "fit",
    "fit": "fit",
    "topuk boyu": "heel_height",
    "heel height": "heel_height",
    "kol tipi": "sleeve_type",
    "yaka tipi": "collar_type",
    "kapama sekli": "closure_type",
    "kapama şekli": "closure_type",
}

COLOR_VALUE_ALIASES: dict[str, str] = {
    "siyah": "black",
    "black": "black",
    "beyaz": "white",
    "white": "white",
    "bej": "beige",
    "beige": "beige",
    "krem": "cream",
    "cream": "cream",
    "lacivert": "navy",
    "navy": "navy",
    "mavi": "blue",
    "blue": "blue",
    "kirmizi": "red",
    "kırmızı": "red",
    "red": "red",
    "yesil": "green",
    "yeşil": "green",
    "green": "green",
    "gri": "gray",
    "grey": "gray",
    "gray": "gray",
    "pembe": "pink",
    "pink": "pink",
    "mor": "purple",
    "purple": "purple",
    "sari": "yellow",
    "sarı": "yellow",
    "yellow": "yellow",
    "kahverengi": "brown",
    "brown": "brown",
    "turuncu": "orange",
    "orange": "orange",
}

MATERIAL_VALUE_ALIASES: dict[str, str] = {
    "deri": "leather",
    "leather": "leather",
    "hakiki deri": "genuine_leather",
    "gercek deri": "genuine_leather",
    "gerçek deri": "genuine_leather",
    "genuine leather": "genuine_leather",
    "suni deri": "faux_leather",
    "sentetik deri": "faux_leather",
    "faux leather": "faux_leather",
    "polyester": "polyester",
    "pamuk": "cotton",
    "cotton": "cotton",
    "keten": "linen",
    "linen": "linen",
    "viskon": "viscose",
    "viscose": "viscose",
    "yün": "wool",
    "yun": "wool",
    "wool": "wool",
}

STYLE_VALUE_ALIASES: dict[str, str] = {
    "spor": "sport",
    "sport": "sport",
    "casual": "casual",
    "klasik": "classic",
    "classic": "classic",
    "modern": "modern",
    "oversize": "oversize",
    "basic": "basic",
    "sik": "chic",
    "şık": "chic",
}

MULTI_VALUE_SPLIT_RE = re.compile(r"\s*(?:/|\||;|\+| ve | & )\s*", flags=re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParsedAttributes:
    attribute_dict: dict[str, list[str]] = field(default_factory=dict)
    attribute_keys: list[str] = field(default_factory=list)
    attribute_values: list[str] = field(default_factory=list)
    normalized_attribute_text: str = ""
    color_value: str = ""
    material_value: str = ""
    style_value: str = ""

    def as_record(self, *, dict_as_json: bool = True) -> dict[str, Any]:
        return {
            SCHEMA.attribute_dict: json.dumps(self.attribute_dict, ensure_ascii=False, sort_keys=True) if dict_as_json else self.attribute_dict,
            SCHEMA.attribute_keys: "|".join(self.attribute_keys),
            SCHEMA.attribute_values: "|".join(self.attribute_values),
            SCHEMA.normalized_attribute_text: self.normalized_attribute_text,
            SCHEMA.color_value: self.color_value,
            SCHEMA.material_value: self.material_value,
            SCHEMA.style_value: self.style_value,
        }


def _strip_accents_for_matching(text: str) -> str:
    # Turkish dotless/dotted i needs explicit handling before NFKD.
    text = text.replace("ı", "i").replace("İ", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_token(text: Any, *, ascii_match: bool = False) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = str(text).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = SPACE_RE.sub(" ", value)
    value = value.strip(" ,.;:()[]{}\"'")
    if ascii_match:
        value = _strip_accents_for_matching(value)
    return SPACE_RE.sub(" ", value).strip()


def _slug(text: str) -> str:
    ascii_text = _strip_accents_for_matching(text)
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    return ascii_text or "unknown"


def normalize_key(raw_key: Any) -> str:
    key = normalize_token(raw_key)
    key_ascii = normalize_token(raw_key, ascii_match=True)
    return KEY_ALIASES.get(key) or KEY_ALIASES.get(key_ascii) or _slug(key_ascii or key)


def _normalize_value_by_key(canonical_key: str, raw_value: Any) -> str:
    value = normalize_token(raw_value)
    value_ascii = normalize_token(raw_value, ascii_match=True)
    if not value:
        return ""
    if canonical_key == "color":
        return COLOR_VALUE_ALIASES.get(value) or COLOR_VALUE_ALIASES.get(value_ascii) or _slug(value_ascii or value)
    if canonical_key == "material":
        # Preserve semantic differences: leather, genuine_leather and faux_leather
        # intentionally remain distinct.
        return MATERIAL_VALUE_ALIASES.get(value) or MATERIAL_VALUE_ALIASES.get(value_ascii) or _slug(value_ascii or value)
    if canonical_key == "style":
        return STYLE_VALUE_ALIASES.get(value) or STYLE_VALUE_ALIASES.get(value_ascii) or _slug(value_ascii or value)
    return _slug(value_ascii or value)


def _split_values(raw_value: str) -> list[str]:
    value = str(raw_value).strip()
    if not value:
        return []
    parts = [part.strip() for part in MULTI_VALUE_SPLIT_RE.split(value) if part.strip()]
    return parts or [value]


def _split_attribute_fragments(raw_attributes: str) -> list[str]:
    """Split key-value fragments on commas.

    The official format is comma-separated. If a value itself contains commas,
    this simple parser may split it; such fragments are preserved as malformed
    `unknown` text rather than discarded.
    """
    return [fragment.strip() for fragment in str(raw_attributes).split(",") if fragment.strip()]


def _add_value(target: dict[str, list[str]], key: str, value: str) -> None:
    if not value:
        return
    values = target.setdefault(key, [])
    if value not in values:
        values.append(value)


def parse_attributes(raw_attributes: Any) -> ParsedAttributes:
    """Parse one raw attributes cell into canonical normalized representations."""
    if raw_attributes is None or (isinstance(raw_attributes, float) and pd.isna(raw_attributes)):
        return ParsedAttributes()

    raw = str(raw_attributes).strip()
    if not raw or raw.lower() in {"nan", "none", "null", "-"}:
        return ParsedAttributes()

    parsed: dict[str, list[str]] = {}
    malformed: list[str] = []

    for fragment in _split_attribute_fragments(raw):
        if ":" not in fragment:
            normalized_fragment = _slug(normalize_token(fragment, ascii_match=True))
            if normalized_fragment and normalized_fragment != "unknown":
                malformed.append(normalized_fragment)
            continue
        raw_key, raw_value = fragment.split(":", 1)
        key = normalize_key(raw_key)
        values = _split_values(raw_value)
        if not values:
            continue
        for value in values:
            normalized_value = _normalize_value_by_key(key, value)
            _add_value(parsed, key, normalized_value)

    for value in malformed:
        _add_value(parsed, "unknown", value)

    keys = sorted(parsed.keys())
    values_flat: list[str] = []
    text_parts: list[str] = []
    for key in keys:
        values = parsed[key]
        values_flat.extend(values)
        text_parts.append(f"{key}: {' '.join(values)}")

    return ParsedAttributes(
        attribute_dict=parsed,
        attribute_keys=keys,
        attribute_values=values_flat,
        normalized_attribute_text=" ; ".join(text_parts),
        color_value="|".join(parsed.get("color", [])),
        material_value="|".join(parsed.get("material", [])),
        style_value="|".join(parsed.get("style", [])),
    )


def add_attribute_features(df: pd.DataFrame, source_col: str | None = None) -> pd.DataFrame:
    """Add normalized attribute feature columns to a dataframe.

    `attribute_dict` is stored as deterministic JSON text to keep CSV/parquet
    output stable across environments. Use `parse_attribute_dict_json` to convert
    it back to a Python dictionary when needed.
    """
    source_col = source_col or SCHEMA.attributes
    if source_col not in df.columns:
        raise KeyError(f"attributes source column not found: {source_col}")
    out = df.copy()
    records = [parse_attributes(value).as_record(dict_as_json=True) for value in out[source_col].tolist()]
    features = pd.DataFrame.from_records(records, index=out.index)
    for col in features.columns:
        out[col] = features[col]
    return out


def parse_attribute_dict_json(value: Any) -> dict[str, list[str]]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return {}
    try:
        data = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [str(vv) for vv in v] if isinstance(v, list) else [str(v)] for k, v in data.items()}
