"""
Tüm feature extractor'ları birleştiren pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.features.bm25_features import BM25Extractor
from src.features.brand_features import BrandFeatureExtractor


class FeaturePipeline:
    NUMERIC_FEATURES = [
        "fuzzy_brand_match_score",
        "has_brand_match",
        "jaccard_similarity",
        "longest_common_substring",
        "bm25_score",
        "text_overlap_ratio",
        "query_coverage",
        # Trendyol DNA (Aşama 4)
        "trendyol_embed_cosine",
        "trendyol_embed_dot",
        "trendyol_embed_q_norm",
        "trendyol_embed_p_norm",
        "trendyol_cos_d128",
        "trendyol_cos_d512",
        "trendyol_cos_d768",
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.brand = BrandFeatureExtractor()
        self.bm25 = BM25Extractor()
        self._fitted = False
        # Trendyol DNA entegrasyonu (Aşama 4)
        self.trendyol_embedder = None
        if config.get("features", {}).get("use_trendyol_embed", True):
            try:
                from src.features.embedding_features import TrendyolEmbedder
                self.trendyol_embedder = TrendyolEmbedder()
            except Exception as e:
                print(f"[FeaturePipeline] TrendyolEmbedder başlatılamadı: {e}")

    def _text_overlap(self, query: str, text: str) -> Tuple[float, float]:
        q_tokens = set(str(query).lower().split())
        t_tokens = set(str(text).lower().split())
        if not q_tokens or not t_tokens:
            return 0.0, 0.0
        overlap = len(q_tokens & t_tokens)
        return overlap / len(q_tokens), overlap / len(t_tokens)

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self.brand.transform(out)
        out = self.bm25.transform(out)

        q_col = self.config.get("data", {}).get("query_column", "search_query")
        t_col = self.config.get("data", {}).get("product_text_column", "product_name")

        # Trendyol DNA embedding features (Aşama 4)
        if self.trendyol_embedder is not None:
            try:
                out = self.trendyol_embedder.cosine_features(out, q_col, t_col)
                out = self.trendyol_embedder.matryoshka_features(out, q_col, t_col)
            except Exception as e:
                print(f"[FeaturePipeline] Trendyol embed feature üretilemedi: {e}")

        overlaps, coverages = [], []
        for _, row in out.iterrows():
            o, c = self._text_overlap(row.get(q_col, ""), row.get(t_col, ""))
            overlaps.append(o)
            coverages.append(c)
        out["text_overlap_ratio"] = overlaps
        out["query_coverage"] = coverages
        return out

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.bm25.fit(df)
        self._fitted = True
        enriched = self._enrich(df)
        cols = [c for c in self.NUMERIC_FEATURES if c in enriched.columns]
        return enriched[cols].fillna(0).values.astype(np.float32), enriched

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            self.bm25.fit(df)
            self._fitted = True
        enriched = self._enrich(df)
        cols = [c for c in self.NUMERIC_FEATURES if c in enriched.columns]
        return enriched[cols].fillna(0).values.astype(np.float32)

    def get_feature_dict(self, df: pd.DataFrame) -> Dict[str, float]:
        enriched = self._enrich(df)
        if len(enriched) == 0:
            return {}
        row = enriched.iloc[0]
        return {c: float(row.get(c, 0.0)) for c in self.NUMERIC_FEATURES if c in enriched.columns}
