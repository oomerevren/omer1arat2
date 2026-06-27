"""Item/query text representation builders for retrieval.

BM25, dense retrieval and cross-encoder training have different text needs.  This
module keeps the representations explicit so a dense index can be reproduced from
metadata instead of from an implicit ``full_item_text`` side effect.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

from src_kaggle.data.attribute_parser import add_attribute_features
from src_kaggle.data.schema import SCHEMA

TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = unicodedata.normalize("NFKC", str(text)).lower()
    return SPACE_RE.sub(" ", value).strip()


def tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def _leaf_category(category: Any) -> str:
    text = normalize_text(category)
    if not text:
        return ""
    for sep in [">", "/", "|", "\\"]:
        if sep in text:
            return normalize_text(text.split(sep)[-1])
    return text


def _dedup_join(parts: list[Any]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        s = normalize_text(p)
        if s and s not in seen:
            out.append(s); seen.add(s)
    return " ".join(out)


def build_item_retrieval_text(row: pd.Series | dict) -> str:
    get = row.get if isinstance(row, dict) else row.get
    parts = [
        get(SCHEMA.brand, ""), get(SCHEMA.title, ""), get(SCHEMA.category, ""),
        f"gender {get(SCHEMA.gender, '')}", f"age {get(SCHEMA.age_group, '')}",
        get(SCHEMA.normalized_attribute_text, get(SCHEMA.attributes, "")),
        get(SCHEMA.color_value, ""), get(SCHEMA.material_value, ""), get(SCHEMA.style_value, ""),
    ]
    return _dedup_join(parts)


def build_item_dense_text_v1(row: pd.Series | dict) -> str:
    """Dense text v1: balanced product card representation."""
    get = row.get if isinstance(row, dict) else row.get
    parts = [
        f"ürün {get(SCHEMA.title, '')}",
        f"kategori {get(SCHEMA.category, '')}",
        f"marka {get(SCHEMA.brand, '')}",
        f"özellik {get(SCHEMA.normalized_attribute_text, get(SCHEMA.attributes, ''))}",
        f"cinsiyet {get(SCHEMA.gender, '')}",
        f"yaş {get(SCHEMA.age_group, '')}",
    ]
    return _dedup_join(parts)


def build_item_dense_text_v2(row: pd.Series | dict) -> str:
    """Dense text v2: title/leaf-category/canonical attributes, less noise."""
    get = row.get if isinstance(row, dict) else row.get
    attrs = [get(SCHEMA.color_value, ""), get(SCHEMA.material_value, ""), get(SCHEMA.style_value, "")]
    parts = [
        get(SCHEMA.title, ""),
        _leaf_category(get(SCHEMA.category, "")),
        get(SCHEMA.brand, ""),
        " ".join(str(x) for x in attrs if str(x).strip()),
        get(SCHEMA.gender, ""),
        get(SCHEMA.age_group, ""),
    ]
    return _dedup_join(parts)


def build_item_dense_text(row: pd.Series | dict, version: str = "dense_v1") -> str:
    if version == "dense_v2":
        return build_item_dense_text_v2(row)
    if version in {"dense_v1", "default", "v1"}:
        return build_item_dense_text_v1(row)
    raise ValueError(f"Unknown dense item text version: {version}")


def build_query_dense_text(query: Any, version: str = "raw") -> str:
    """Safe query representation; intentionally not over-expanded."""
    return normalize_text(query)


def prepare_items_for_retrieval(items: pd.DataFrame, dense_text_version: str = "dense_v1") -> pd.DataFrame:
    out = add_attribute_features(items) if SCHEMA.normalized_attribute_text not in items.columns else items.copy()
    out[SCHEMA.retrieval_text] = out.apply(build_item_retrieval_text, axis=1)
    out[SCHEMA.dense_text] = out.apply(lambda r: build_item_dense_text(r, dense_text_version), axis=1)
    if SCHEMA.full_item_text not in out.columns:
        out[SCHEMA.full_item_text] = out[SCHEMA.retrieval_text]
    return out.reset_index(drop=True)
