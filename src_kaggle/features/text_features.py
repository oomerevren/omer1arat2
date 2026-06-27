"""Text feature builder for canonical TEKNOFEST Kaggle schema."""

from __future__ import annotations

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from src_kaggle.data.schema import SCHEMA


class PairTextVectorizer:
    """TF-IDF representation over query, item fields and combined pair text."""

    def __init__(self, max_features: int = 200_000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.query_vectorizer = TfidfVectorizer(max_features=max_features // 4, ngram_range=ngram_range, min_df=1)
        self.item_vectorizer = TfidfVectorizer(max_features=max_features // 2, ngram_range=ngram_range, min_df=1)
        self.pair_vectorizer = TfidfVectorizer(max_features=max_features // 4, ngram_range=ngram_range, min_df=1)

    def _item_text(self, df: pd.DataFrame) -> pd.Series:
        if SCHEMA.full_item_text in df.columns:
            return df[SCHEMA.full_item_text].fillna("").astype(str)
        parts = [
            df[SCHEMA.title].fillna(""),
            df[SCHEMA.category].fillna(""),
            df[SCHEMA.brand].fillna(""),
            df[SCHEMA.gender].fillna("unknown"),
            df[SCHEMA.age_group].fillna("unknown"),
            df.get(SCHEMA.normalized_attribute_text, df[SCHEMA.attributes]).fillna(""),
        ]
        return pd.concat(parts, axis=1).astype(str).agg(" ".join, axis=1).str.strip()

    def _texts(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        query = df[SCHEMA.query].fillna("").astype(str)
        item = self._item_text(df)
        pair = query + " [SEP] " + item
        return query, item, pair

    def fit_transform(self, df: pd.DataFrame):
        query, item, pair = self._texts(df)
        return hstack([
            self.query_vectorizer.fit_transform(query),
            self.item_vectorizer.fit_transform(item),
            self.pair_vectorizer.fit_transform(pair),
        ]).tocsr()

    def transform(self, df: pd.DataFrame):
        query, item, pair = self._texts(df)
        return hstack([
            self.query_vectorizer.transform(query),
            self.item_vectorizer.transform(item),
            self.pair_vectorizer.transform(pair),
        ]).tocsr()
