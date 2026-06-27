"""Candidate pool builders for multi-layer negative mining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src_kaggle.data.attribute_parser import add_attribute_features
from src_kaggle.data.pair_builder import build_full_item_text
from src_kaggle.data.schema import SCHEMA
from src_kaggle.features.query_intent import build_query_intent_resources, extract_query_intent


@dataclass
class CandidatePool:
    items: pd.DataFrame
    vectorizer: TfidfVectorizer
    item_matrix: any
    rng: np.random.Generator

    @classmethod
    def from_items(cls, items: pd.DataFrame, seed: int = 42, max_features: int = 200_000) -> "CandidatePool":
        items = add_attribute_features(items) if SCHEMA.normalized_attribute_text not in items.columns else items.copy()
        items = build_full_item_text(items)
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
        item_matrix = vectorizer.fit_transform(items[SCHEMA.full_item_text].fillna("").astype(str))
        return cls(items=items.reset_index(drop=True), vectorizer=vectorizer, item_matrix=item_matrix, rng=np.random.default_rng(seed))

    def lexical_scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query or ""])
        return cosine_similarity(q, self.item_matrix).ravel()

    def top_lexical(self, query: str, k: int, exclude_items: set) -> pd.DataFrame:
        scores = self.lexical_scores(query)
        order = np.argsort(-scores)
        rows = []
        for idx in order:
            item_id = self.items.iloc[idx][SCHEMA.item_id]
            if item_id in exclude_items:
                continue
            row = self.items.iloc[idx].copy()
            row["lexical_score"] = float(scores[idx])
            rows.append(row)
            if len(rows) >= k:
                break
        return pd.DataFrame(rows)

    def random_far(self, query: str, k: int, exclude_items: set, max_lexical: float = 0.08) -> pd.DataFrame:
        scores = self.lexical_scores(query)
        eligible = [i for i, row in self.items.iterrows() if row[SCHEMA.item_id] not in exclude_items and scores[i] <= max_lexical]
        if not eligible:
            eligible = [i for i, row in self.items.iterrows() if row[SCHEMA.item_id] not in exclude_items]
        if not eligible:
            return pd.DataFrame()
        chosen = self.rng.choice(eligible, size=min(k, len(eligible)), replace=False)
        out = self.items.iloc[list(chosen)].copy()
        out["lexical_score"] = [float(scores[i]) for i in chosen]
        return out

    def same_category(self, positive_categories: set[str], query: str, k: int, exclude_items: set) -> pd.DataFrame:
        scores = self.lexical_scores(query)
        mask = self.items[SCHEMA.category].astype(str).isin(positive_categories)
        rows = self.items[mask].copy()
        rows = rows[~rows[SCHEMA.item_id].isin(exclude_items)]
        if rows.empty:
            return rows
        rows["lexical_score"] = [float(scores[i]) for i in rows.index]
        return rows.sort_values("lexical_score", ascending=False).head(k)

    def same_brand(self, positive_brands: set[str], query: str, k: int, exclude_items: set) -> pd.DataFrame:
        scores = self.lexical_scores(query)
        mask = self.items[SCHEMA.brand].astype(str).isin(positive_brands)
        rows = self.items[mask].copy()
        rows = rows[~rows[SCHEMA.item_id].isin(exclude_items)]
        if rows.empty:
            return rows
        rows["lexical_score"] = [float(scores[i]) for i in rows.index]
        return rows.sort_values("lexical_score", ascending=False).head(k)

    def attribute_conflict(self, query: str, k: int, exclude_items: set) -> pd.DataFrame:
        resources = build_query_intent_resources(self.items)
        intent = extract_query_intent(query, resources)
        scores = self.lexical_scores(query)
        rows = self.items[~self.items[SCHEMA.item_id].isin(exclude_items)].copy()
        if rows.empty:
            return rows
        q_colors = set(filter(None, str(intent.get("detected_color_candidates", "")).split("|")))
        q_materials = set(filter(None, str(intent.get("detected_material_candidates", "")).split("|")))
        conflict_mask = pd.Series(False, index=rows.index)
        if q_colors and SCHEMA.color_value in rows.columns:
            conflict_mask |= rows[SCHEMA.color_value].fillna("").astype(str).map(lambda x: bool(x) and q_colors.isdisjoint(set(filter(None, x.split("|")))))
        if q_materials and SCHEMA.material_value in rows.columns:
            conflict_mask |= rows[SCHEMA.material_value].fillna("").astype(str).map(lambda x: bool(x) and q_materials.isdisjoint(set(filter(None, x.split("|")))))
        rows = rows[conflict_mask]
        if rows.empty:
            return rows
        rows["lexical_score"] = [float(scores[i]) for i in rows.index]
        return rows.sort_values("lexical_score", ascending=False).head(k)
