import random
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class NegativeMiner:
    """
    Multi-stage Hard Negative Mining.
    Stratejiler: random, in-category, bm25_hard, embedding_nearby, same_brand.
    """

    def __init__(
        self,
        negatives_per_positive: int = 5,
        strategy: str = "mixed",
        strategies: Optional[Dict[str, int]] = None,
        dense_model_name: Optional[str] = None,
    ):
        self.n_neg = negatives_per_positive
        self.strategy = strategy
        self.strategies = strategies or {}
        self.vectorizer = None
        self.tfidf_matrix = None
        self._product_embeddings = None
        self.dense_model_name = dense_model_name
        self.dense_model = None
        self.dense_embeddings = None

    def _fit_dense(self, df: pd.DataFrame) -> None:
        """Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 ile dense embedding (Aşama 4)."""
        try:
            from src.features.embedding_features import TrendyolEmbedder
            self.embedder = TrendyolEmbedder(batch_size=32)
            corpus = (
                df["product_name"].fillna("") + " | "
                + df.get("category", pd.Series(dtype=str)).fillna("") + " | "
                + df.get("brand", pd.Series(dtype=str)).fillna("")
            ).tolist()
            self.dense_embeddings = self.embedder.encode(corpus)
        except Exception as exc:
            print(f"[NegativeMiner] TrendyolEmbedder başlatılamadı: {exc}. TF-IDF fallback kullanılacak.")
            self.embedder = None
            self.dense_embeddings = None

    def _fit_tfidf(self, df: pd.DataFrame) -> None:
        if not HAS_SKLEARN:
            return
        corpus = (
            df["product_name"].fillna("")
            + " "
            + df.get("category", pd.Series(dtype=str)).fillna("")
        ).tolist()
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self._product_embeddings = self.tfidf_matrix

    def _mine_random(
        self, query: str, true_product_id: str, pool_df: pd.DataFrame, n: int
    ) -> pd.DataFrame:
        candidates = pool_df[pool_df["product_id"] != true_product_id]
        if len(candidates) > n:
            return candidates.sample(n, random_state=42)
        return candidates

    def _mine_in_category(
        self, query: str, true_product_id: str, true_cat: str, pool_df: pd.DataFrame, n: int
    ) -> pd.DataFrame:
        if not true_cat or "category" not in pool_df.columns:
            return self._mine_random(query, true_product_id, pool_df, n)
        candidates = pool_df[
            (pool_df["category"] == true_cat) & (pool_df["product_id"] != true_product_id)
        ]
        if len(candidates) > n:
            return candidates.sample(n, random_state=42)
        return candidates

    def _mine_same_brand(
        self, query: str, true_product_id: str, brand: str, pool_df: pd.DataFrame, n: int
    ) -> pd.DataFrame:
        if not brand or "brand" not in pool_df.columns:
            return self._mine_random(query, true_product_id, pool_df, n)
        candidates = pool_df[
            (pool_df["brand"] == brand) & (pool_df["product_id"] != true_product_id)
        ]
        if len(candidates) > n:
            return candidates.sample(n, random_state=42)
        return candidates

    def _mine_similarity_hard(
        self, query: str, true_product_id: str, pool_df: pd.DataFrame, n: int
    ) -> pd.DataFrame:
        if not HAS_SKLEARN or self.vectorizer is None:
            return self._mine_random(query, true_product_id, pool_df, n)

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1]

        selected_idx = []
        for idx in top_indices:
            if pool_df.iloc[idx]["product_id"] != true_product_id:
                selected_idx.append(idx)
            if len(selected_idx) == n:
                break
        return pool_df.iloc[selected_idx] if selected_idx else pool_df.head(0)

    def _mine_embedding_nearby(
        self, query: str, true_product_id: str, pool_df: pd.DataFrame, n: int
    ) -> pd.DataFrame:
        """Dense embedding yakınlığı ile hard negative. Yoksa TF-IDF fallback."""
        if self.embedder is not None and self.dense_embeddings is not None:
            query_emb = self.embedder.encode([query])
            # Embeddings normalize=True → dot product = cosine similarity
            sims = np.dot(query_emb, self.dense_embeddings.T)[0]
            top_indices = np.argsort(sims)[::-1]

            selected_idx = []
            for idx in top_indices:
                if pool_df.iloc[idx]["product_id"] != true_product_id:
                    selected_idx.append(idx)
                if len(selected_idx) == n:
                    break
            return pool_df.iloc[selected_idx] if selected_idx else pool_df.head(0)
        else:
            return self._mine_similarity_hard(query, true_product_id, pool_df, n)

    def _allocate_quotas(self) -> Dict[str, int]:
        if self.strategies:
            return {
                "random": self.strategies.get("random", 0),
                "in_category": self.strategies.get("same_category", 0),
                "bm25_hard": self.strategies.get("bm25_hard", 0),
                "embedding_nearby": self.strategies.get("embedding_nearby", 0),
                "same_brand": self.strategies.get("same_brand", 0),
            }
        n_rand = max(1, self.n_neg // 3)
        n_cat = max(1, self.n_neg // 3)
        n_hard = self.n_neg - n_rand - n_cat
        return {
            "random": n_rand,
            "in_category": n_cat,
            "bm25_hard": n_hard // 2,
            "embedding_nearby": n_hard - n_hard // 2,
            "same_brand": 0,
        }

    def mine(self, df: pd.DataFrame) -> pd.DataFrame:
        if "product_id" not in df.columns or "search_query" not in df.columns:
            return df

        label_col = "is_relevant"
        positives = df[df.get(label_col, 1) >= 1] if label_col in df.columns else df
        if len(positives) == 0:
            positives = df

        pool_df = df.copy()
        self._fit_tfidf(pool_df)
        self._fit_dense(pool_df)

        augmented_rows = []
        quotas = self._allocate_quotas()
        total_quota = sum(quotas.values()) or self.n_neg

        for _, row in positives.iterrows():
            pos_row = row.copy()
            if label_col not in pos_row or pd.isna(pos_row[label_col]):
                pos_row[label_col] = 1
            augmented_rows.append(pos_row)

            query = row["search_query"]
            pid = row["product_id"]
            cat = row.get("category", "")
            brand = row.get("brand", "")

            scale = self.n_neg / max(total_quota, 1)
            mined = []
            mined.extend(
                self._mine_random(query, pid, pool_df, max(0, int(quotas["random"] * scale))).to_dict("records")
            )
            mined.extend(
                self._mine_in_category(query, pid, cat, pool_df, max(0, int(quotas["in_category"] * scale))).to_dict("records")
            )
            mined.extend(
                self._mine_similarity_hard(query, pid, pool_df, max(0, int(quotas["bm25_hard"] * scale))).to_dict("records")
            )
            mined.extend(
                self._mine_embedding_nearby(query, pid, pool_df, max(0, int(quotas["embedding_nearby"] * scale))).to_dict("records")
            )
            mined.extend(
                self._mine_same_brand(query, pid, brand, pool_df, max(0, int(quotas["same_brand"] * scale))).to_dict("records")
            )

            seen = set()
            for neg in mined:
                key = (query, neg.get("product_id"))
                if key in seen or neg.get("product_id") == pid:
                    continue
                seen.add(key)
                new_row = pd.Series(neg).copy()
                new_row["search_query"] = query
                new_row[label_col] = 0
                augmented_rows.append(new_row)

        result_df = pd.DataFrame(augmented_rows)
        result_df = result_df.drop_duplicates(subset=["search_query", "product_id"])
        return result_df.sample(frac=1.0, random_state=42).reset_index(drop=True)


HardNegativeMinerV2 = NegativeMiner
