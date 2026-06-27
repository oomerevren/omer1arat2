"""Rule-based query intent and segmentation extractor for Kaggle War Mode.

The goal is explainable segmentation, not black-box understanding. Signals from
this module are intended for feature engineering, segment-level validation,
error analysis, threshold tuning and negative sampling strategy.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from src_kaggle.data.attribute_parser import (
    COLOR_VALUE_ALIASES,
    MATERIAL_VALUE_ALIASES,
    STYLE_VALUE_ALIASES,
    normalize_token,
    parse_attributes,
)
from src_kaggle.data.schema import SCHEMA

TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}", flags=re.IGNORECASE)
SIZE_NUMBER_RE = re.compile(r"^(?:[1-9][0-9]|[0-9]{2,3})$")
SIZE_ALPHA_RE = re.compile(r"^(?:xxs|xs|s|m|l|xl|xxl|xxxl|\d+xl)$", flags=re.IGNORECASE)

GENDER_TERMS: dict[str, str] = {
    "kadin": "female",
    "kadın": "female",
    "bayan": "female",
    "kiz": "female_child",
    "kız": "female_child",
    "erkek": "male",
    "bay": "male",
    "unisex": "unisex",
}

AGE_TERMS: dict[str, str] = {
    "bebek": "baby",
    "yenidogan": "newborn",
    "yenidoğan": "newborn",
    "cocuk": "child",
    "çocuk": "child",
    "cocuklar": "child",
    "çocuklar": "child",
    "genc": "teen",
    "genç": "teen",
    "yetiskin": "adult",
    "yetişkin": "adult",
    "okul": "school_age",
    "anaokulu": "preschool",
}

SIZE_CUE_TERMS = {
    "beden",
    "numara",
    "no",
    "yas",
    "yaş",
    "ay",
    "cm",
    "kg",
    "small",
    "medium",
    "large",
}

# Category-ish high-signal words independent from item metadata. Metadata tokens
# are added by QueryIntentResources.from_items.
BASE_CATEGORY_TERMS = {
    "sneaker",
    "ayakkabi",
    "ayakkabı",
    "ayakkabisi",
    "ayakkabısı",
    "ceket",
    "mont",
    "canta",
    "çanta",
    "elbise",
    "gomlek",
    "gömlek",
    "pantolon",
    "sweatshirt",
    "tisort",
    "tişört",
    "zibin",
    "zıbın",
    "body",
    "bot",
    "kazak",
    "etek",
    "sort",
    "şort",
    "tayt",
    "corap",
    "çorap",
}

ATTRIBUTE_KEY_TERMS = {
    "renk",
    "color",
    "materyal",
    "malzeme",
    "kumas",
    "kumaş",
    "stil",
    "style",
    "desen",
    "beden",
    "kalip",
    "kalıp",
}


@dataclass(frozen=True)
class QueryIntentResources:
    """Dictionaries used by the rule-based query segmenter."""

    brand_tokens: set[str] = field(default_factory=set)
    category_tokens: set[str] = field(default_factory=lambda: set(BASE_CATEGORY_TERMS))
    color_tokens: set[str] = field(default_factory=set)
    material_tokens: set[str] = field(default_factory=set)
    style_tokens: set[str] = field(default_factory=set)
    attribute_tokens: set[str] = field(default_factory=set)

    @staticmethod
    def from_items(items: pd.DataFrame | None) -> "QueryIntentResources":
        if items is None or items.empty:
            return default_query_intent_resources()

        brand_tokens: set[str] = set()
        category_tokens: set[str] = set(BASE_CATEGORY_TERMS)
        attr_tokens: set[str] = set(ATTRIBUTE_KEY_TERMS)
        color_tokens = _alias_tokens(COLOR_VALUE_ALIASES)
        material_tokens = _alias_tokens(MATERIAL_VALUE_ALIASES)
        style_tokens = _alias_tokens(STYLE_VALUE_ALIASES)

        if SCHEMA.brand in items.columns:
            for brand in items[SCHEMA.brand].dropna().astype(str).unique():
                toks = tokenize(brand)
                if toks:
                    brand_tokens.add(" ".join(toks))
                    brand_tokens.update(toks)

        if SCHEMA.category in items.columns:
            for category in items[SCHEMA.category].dropna().astype(str).unique():
                category_tokens.update(tokenize(category))

        if SCHEMA.attributes in items.columns:
            for raw in items[SCHEMA.attributes].dropna().astype(str).head(200_000):
                parsed = parse_attributes(raw)
                attr_tokens.update(parsed.attribute_keys)
                color_tokens.update(_split_candidate_values(parsed.color_value))
                material_tokens.update(_split_candidate_values(parsed.material_value))
                style_tokens.update(_split_candidate_values(parsed.style_value))

        return QueryIntentResources(
            brand_tokens={t for t in brand_tokens if t},
            category_tokens={_norm_for_match(t) for t in category_tokens if t},
            color_tokens={_norm_for_match(t) for t in color_tokens if t},
            material_tokens={_norm_for_match(t) for t in material_tokens if t},
            style_tokens={_norm_for_match(t) for t in style_tokens if t},
            attribute_tokens={_norm_for_match(t) for t in attr_tokens if t},
        )


def _strip_accents(text: str) -> str:
    text = text.replace("ı", "i").replace("İ", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _norm_for_match(text: Any) -> str:
    return _strip_accents(normalize_token(text)).strip()


def tokenize(query: Any) -> list[str]:
    if query is None or (isinstance(query, float) and pd.isna(query)):
        return []
    text = _norm_for_match(query)
    return TOKEN_RE.findall(text)


def _alias_tokens(alias_map: dict[str, str]) -> set[str]:
    out: set[str] = set()
    for raw, normalized in alias_map.items():
        out.add(_norm_for_match(raw))
        out.add(_norm_for_match(normalized))
    return out


def _split_candidate_values(value: str) -> set[str]:
    if not value:
        return set()
    return {_norm_for_match(v) for v in str(value).split("|") if str(v).strip()}


def default_query_intent_resources() -> QueryIntentResources:
    return QueryIntentResources(
        brand_tokens=set(),
        category_tokens={_norm_for_match(t) for t in BASE_CATEGORY_TERMS},
        color_tokens=_alias_tokens(COLOR_VALUE_ALIASES),
        material_tokens=_alias_tokens(MATERIAL_VALUE_ALIASES),
        style_tokens=_alias_tokens(STYLE_VALUE_ALIASES),
        attribute_tokens={_norm_for_match(t) for t in ATTRIBUTE_KEY_TERMS},
    )


def _detect(tokens: list[str], candidates: set[str]) -> list[str]:
    token_set = set(tokens)
    detected = sorted(token_set & candidates)
    joined = " ".join(tokens)
    # Include multi-token candidates such as "genuine leather" or brand names.
    for candidate in candidates:
        if " " in candidate and candidate in joined and candidate not in detected:
            detected.append(candidate)
    return sorted(detected)


def _has_size(tokens: list[str]) -> bool:
    token_set = set(tokens)
    if token_set & {_norm_for_match(t) for t in SIZE_CUE_TERMS}:
        return True
    return any(SIZE_NUMBER_RE.match(t) or SIZE_ALPHA_RE.match(t) for t in tokens)


def _possible_typo_or_ambiguous(tokens: list[str], known_tokens: set[str]) -> bool:
    if not tokens:
        return True
    if any(REPEATED_CHAR_RE.search(t) for t in tokens):
        return True
    if any(len(t) >= 14 for t in tokens):
        return True
    alpha_tokens = [t for t in tokens if any(ch.isalpha() for ch in t)]
    if not alpha_tokens:
        return False
    # Turkish words normally contain vowels; no-vowel long alpha token is suspicious.
    if any(len(t) >= 5 and not set(t) & set("aeiouöü") for t in alpha_tokens):
        return True
    known_ratio = sum(1 for t in alpha_tokens if t in known_tokens) / max(1, len(alpha_tokens))
    return len(alpha_tokens) <= 2 and known_ratio == 0


def extract_query_intent(query: Any, resources: QueryIntentResources | None = None) -> dict[str, Any]:
    """Extract explainable query intent features for one query."""
    resources = resources or default_query_intent_resources()
    tokens = tokenize(query)
    token_count = len(tokens)

    brands = _detect(tokens, resources.brand_tokens)
    categories = _detect(tokens, resources.category_tokens)
    colors = _detect(tokens, resources.color_tokens)
    materials = _detect(tokens, resources.material_tokens)
    styles = _detect(tokens, resources.style_tokens)
    genders_raw = _detect(tokens, {_norm_for_match(t) for t in GENDER_TERMS})
    ages_raw = _detect(tokens, {_norm_for_match(t) for t in AGE_TERMS})

    attribute_signal_count = len(colors) + len(materials) + len(styles)
    category_signal_count = len(categories)
    brand_signal_count = len(brands)

    known_tokens = set().union(
        resources.brand_tokens,
        resources.category_tokens,
        resources.color_tokens,
        resources.material_tokens,
        resources.style_tokens,
        resources.attribute_tokens,
        {_norm_for_match(t) for t in GENDER_TERMS},
        {_norm_for_match(t) for t in AGE_TERMS},
        {_norm_for_match(t) for t in SIZE_CUE_TERMS},
    )

    return {
        "query_normalized": " ".join(tokens),
        "query_length_tokens": token_count,
        "query_char_length": len(str(query)) if query is not None else 0,
        "has_brand_token": int(bool(brands)),
        "has_category_token": int(bool(categories)),
        "has_color_token": int(bool(colors)),
        "has_material_token": int(bool(materials)),
        "has_style_token": int(bool(styles)),
        "has_gender_token": int(bool(genders_raw)),
        "has_age_token": int(bool(ages_raw)),
        "has_size_token": int(_has_size(tokens)),
        "is_short_query": int(token_count <= 2),
        "is_long_query": int(token_count >= 5),
        "is_attribute_heavy": int(attribute_signal_count >= 2 or (token_count > 0 and attribute_signal_count / token_count >= 0.5)),
        "is_brand_heavy": int(bool(brands) and (token_count <= 3 or brand_signal_count / max(1, token_count) >= 0.34)),
        "is_category_heavy": int(bool(categories) and (category_signal_count / max(1, token_count) >= 0.34)),
        "possible_typo_or_ambiguous": int(_possible_typo_or_ambiguous(tokens, known_tokens)),
        "detected_brand_candidates": "|".join(brands),
        "detected_category_candidates": "|".join(categories),
        "detected_color_candidates": "|".join(colors),
        "detected_material_candidates": "|".join(materials),
        "detected_style_candidates": "|".join(styles),
        "detected_gender_candidates": "|".join(GENDER_TERMS.get(g, g) for g in genders_raw),
        "detected_age_candidates": "|".join(AGE_TERMS.get(a, a) for a in ages_raw),
    }


def add_query_intent_features(
    df: pd.DataFrame,
    resources: QueryIntentResources | None = None,
    query_col: str = SCHEMA.query,
) -> pd.DataFrame:
    """Add query intent feature columns to a dataframe."""
    if query_col not in df.columns:
        raise KeyError(f"query column not found: {query_col}")
    resources = resources or default_query_intent_resources()
    out = df.copy()
    records = [extract_query_intent(query, resources) for query in out[query_col].tolist()]
    features = pd.DataFrame.from_records(records, index=out.index)
    for col in features.columns:
        out[col] = features[col]
    return out


def build_query_intent_resources(items: pd.DataFrame | None = None) -> QueryIntentResources:
    """Public helper for callers that prefer function style."""
    return QueryIntentResources.from_items(items)
