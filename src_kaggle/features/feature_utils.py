"""Shared utilities for Kaggle feature engineering."""
from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+")
NUM_RE = re.compile(r"\d+")


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def ascii_fold(value: Any) -> str:
    text = normalize_text(value).replace("ı", "i").replace("İ", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokens(value: Any) -> list[str]:
    return TOKEN_RE.findall(ascii_fold(value))


def token_set(value: Any) -> set[str]:
    return set(tokens(value))


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def seq_ratio(a: Any, b: Any) -> float:
    a = ascii_fold(a); b = ascii_fold(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_sort_ratio(a: Any, b: Any) -> float:
    return seq_ratio(" ".join(sorted(tokens(a))), " ".join(sorted(tokens(b))))


def token_set_ratio(a: Any, b: Any) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta and not tb:
        return 1.0
    inter = ta & tb
    if not inter:
        return 0.0
    return max(safe_div(len(inter), len(ta)), safe_div(len(inter), len(tb)), safe_div(2 * len(inter), len(ta) + len(tb)))


def char_ngrams(text: Any, n: int = 3) -> set[str]:
    s = ascii_fold(text).replace(" ", "_")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s)-n+1)}


def longest_common_token_span(a_tokens: list[str], b_tokens: list[str]) -> int:
    if not a_tokens or not b_tokens:
        return 0
    dp = [[0]*(len(b_tokens)+1) for _ in range(len(a_tokens)+1)]
    best = 0
    for i, x in enumerate(a_tokens, 1):
        for j, y in enumerate(b_tokens, 1):
            if x == y:
                dp[i][j] = dp[i-1][j-1] + 1
                best = max(best, dp[i][j])
    return best


def numeric_tokens(value: Any) -> set[str]:
    return set(NUM_RE.findall(normalize_text(value)))


def is_unknown(value: Any) -> bool:
    return normalize_text(value) in {"", "unknown", "nan", "none", "null", "-"}


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out
