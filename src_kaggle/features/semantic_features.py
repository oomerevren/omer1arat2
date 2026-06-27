from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src_kaggle.data.schema import SCHEMA


def _cosine_pairwise(left: list[str], right: list[str]) -> list[float]:
    corpus = left + right
    if not corpus:
        return []
    vec = TfidfVectorizer(max_features=100_000, ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(corpus)
    n = len(left)
    sims=[]
    for i in range(n):
        sims.append(float(cosine_similarity(X[i], X[n+i]).ravel()[0]))
    return sims


def build_semantic_features(df: pd.DataFrame) -> pd.DataFrame:
    q = df[SCHEMA.query].fillna("").astype(str).tolist()
    title = df[SCHEMA.title].fillna("").astype(str).tolist()
    full = df.get(SCHEMA.full_item_text, df[SCHEMA.title]).fillna("").astype(str).tolist()
    cat = df[SCHEMA.category].fillna("").astype(str).tolist()
    return pd.DataFrame({
        "sem_query_title_cosine": _cosine_pairwise(q, title),
        "sem_query_item_full_text_cosine": _cosine_pairwise(q, full),
        "sem_query_category_cosine": _cosine_pairwise(q, cat),
    }, index=df.index)
