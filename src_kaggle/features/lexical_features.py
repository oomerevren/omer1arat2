from __future__ import annotations

import pandas as pd

from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.feature_utils import (
    char_ngrams, longest_common_token_span, normalize_text, safe_div, seq_ratio,
    token_set, token_set_ratio, token_sort_ratio, tokens,
)


def build_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        q = r.get(SCHEMA.query, ""); title = r.get(SCHEMA.title, "")
        qt, tt = tokens(q), tokens(title)
        qs, ts = set(qt), set(tt)
        inter = qs & ts
        q_norm, t_norm = normalize_text(q), normalize_text(title)
        q3, t3 = char_ngrams(q, 3), char_ngrams(title, 3)
        first_match = 0
        if qt and tt:
            first_match = int(qt[0] in tt[:2])
        rows.append({
            "lex_token_overlap_count": len(inter),
            "lex_token_overlap_ratio": safe_div(len(inter), len(qs | ts)),
            "lex_query_coverage_ratio": safe_div(len(inter), len(qs)),
            "lex_title_coverage_ratio": safe_div(len(inter), len(ts)),
            "lex_unique_query_tokens": len(qs),
            "lex_unique_title_tokens": len(ts),
            "lex_char3_jaccard": safe_div(len(q3 & t3), len(q3 | t3)),
            "lex_longest_common_token_span": longest_common_token_span(qt, tt),
            "lex_longest_common_token_span_ratio": safe_div(longest_common_token_span(qt, tt), len(qt)),
            "lex_exact_phrase_match_flag": int(bool(q_norm) and q_norm in t_norm),
            "lex_partial_substring_match_flag": int(any(len(tok) >= 4 and tok in t_norm for tok in qt)),
            "lex_first_token_early_match_flag": first_match,
            "lex_fuzzy_ratio": seq_ratio(q, title),
            "lex_token_sort_ratio": token_sort_ratio(q, title),
            "lex_token_set_ratio": token_set_ratio(q, title),
        })
    return pd.DataFrame(rows, index=df.index)
