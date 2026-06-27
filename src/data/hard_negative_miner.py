"""
Hard Negative Mining — Aşama 8 (KDD Cup 2022 + Trendyol pattern).

Trendyol train setinde yalnızca alakalı çiftler var → negatif örnek mining KRİTİK.
3 seviye negatif:
  1. Easy: Farklı kategori, farklı marka
  2. Hard: Aynı kategori, benzer embedding cosine (bi-encoder skoru yüksek ama alakasız)
  3. Confounding: Aynı kategori, farklı özellik (örn: L beden vs M beden)

Kullanım:
    miner = HardNegativeMiner(config)
    hard_pairs = miner.mine(train_df, all_products_df)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import mine_hard_negatives
    HAS_ST = True
except ImportError:
    HAS_ST = False
    mine_hard_negatives = None


try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class HardNegativeMiner:
    """
    Aşama 8 — Gelişmiş hard negative mining.
    Trendyol/TY-ecomm-embed veya E5 ile dense embedding yakınlığına göre
    zorlayıcı negatif örnekler üretir.
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        embedder_model: str | None = None,
        num_negatives: int = 8,
        range_min: int = 20,
        range_max: int = 200,
        max_score: float = 0.85,
        margin: float = 0.05,
        use_faiss: bool = True,
        batch_size: int = 4096,
    ):
        self.config = config or {}
        self.embedder_model = embedder_model or self.config.get(
            "features", {}
        ).get("trendyol_embed_model", "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0")
        self.num_negatives = num_negatives
        self.range_min = range_min
        self.range_max = range_max
        self.max_score = max_score
        self.margin = margin
        self.use_faiss = use_faiss and HAS_FAISS
        self.batch_size = batch_size
        self._embedder = None

    def _load_embedder(self) -> Optional[Any]:
        if self._embedder is not None:
            return self._embedder
        if not HAS_ST:
            print("[HardNegativeMiner] sentence-transformers yüklü değil. Fallback.")
            return None
        try:
            self._embedder = SentenceTransformer(
                self.embedder_model,
                trust_remote_code=True,
            )
            return self._embedder
        except Exception as e:
            print(f"[HardNegativeMiner] Embedder yüklenemedi: {e}")
            return None

    def mine(
        self,
        train_df: pd.DataFrame,
        all_products: pd.DataFrame | None = None,
        query_col: str = "search_query",
        product_col: str = "product_name",
        product_id_col: str = "product_id",
        label_col: str = "is_relevant",
    ) -> pd.DataFrame:
        """
        Train setindeki pozitif (query, product) çiftlerinden hard negative üret.

        Args:
            train_df: Sadece pozitif çiftler içeren DataFrame
            all_products: Tüm ürün kataloğu (product_id, product_name).
                         None ise train_df'deki unique ürünler kullanılır.
            query_col: Sorgu sütunu
            product_col: Ürün adı sütunu
            product_id_col: Ürün ID sütunu
            label_col: Etiket sütunu

        Returns:
            DataFrame: Pozitif + negatif çiftler (label = 1/0)
        """
        if label_col in train_df.columns:
            pos_df = train_df[train_df[label_col] == 1].copy()
        else:
            pos_df = train_df.copy()

        if len(pos_df) == 0:
            return train_df

        # Ürün kataloğu
        if all_products is None:
            all_products = pos_df[[product_id_col, product_col]].drop_duplicates(
                subset=[product_id_col]
            ).reset_index(drop=True)

        corpus = all_products[product_col].astype(str).tolist()
        corpus_ids = all_products[product_id_col].astype(str).tolist()

        embedder = self._load_embedder()

        if embedder and mine_hard_negatives is not None:
            # Sentence-Transformers mine_hard_negatives ile mining
            pos_queries = pos_df[query_col].astype(str).tolist()
            pos_products = pos_df[product_col].astype(str).tolist()

            # Sentence-transformers API: mine_hard_negatives(query, corpus, model, ...)
            try:
                hard_negatives = mine_hard_negatives(
                    queries=pos_queries,
                    corpus=corpus,
                    bi_encoder_model=embedder,
                    num_negatives=self.num_negatives,
                    range_min=self.range_min,
                    range_max=self.range_max,
                    max_score=self.max_score,
                    margin=self.margin,
                    use_faiss=self.use_faiss,
                    batch_size=self.batch_size,
                    output_format="labeled-pair",
                )
                # hard_negatives: [(query, neg_product, 0), ...]
                neg_rows = []
                for q, neg_prod, neg_label in hard_negatives:
                    # Orijinal pozitifin diğer kolonlarını bul
                    orig = pos_df[pos_df[query_col] == q].iloc[0]
                    row = orig.copy()
                    row[product_col] = neg_prod
                    # neg ürünün ID'sini bul
                    matching = all_products[all_products[product_col] == neg_prod]
                    if not matching.empty:
                        row[product_id_col] = matching.iloc[0][product_id_col]
                    else:
                        row[product_id_col] = f"neg_{hash(neg_prod) % 10000000}"
                    row[label_col] = int(neg_label)
                    row["negative_type"] = "hard_embedding"
                    neg_rows.append(row)

                neg_df = pd.DataFrame(neg_rows)
                return pd.concat([pos_df, neg_df], ignore_index=True)
            except Exception as e:
                print(f"[HardNegativeMiner] mine_hard_negatives hatası: {e}. Fallback.")

        # Fallback: BM25 + kategori bazlı basit mining
        return self._fallback_mine(
            pos_df, all_products, query_col, product_col, product_id_col, label_col
        )

    def _fallback_mine(
        self,
        pos_df: pd.DataFrame,
        all_products: pd.DataFrame,
        query_col: str,
        product_col: str,
        product_id_col: str,
        label_col: str,
    ) -> pd.DataFrame:
        """Basit fallback: aynı kategori farklı ürün + rastgele."""
        neg_rows = []
        pos_ids = set(pos_df[product_id_col].astype(str))
        for _, row in pos_df.iterrows():
            query = row[query_col]
            # Aynı kategori (varsa) farklı ürün
            cat_val = row.get("category", "")
            if "category" in all_products.columns and cat_val:
                cat_candidates = all_products[
                    (all_products["category"] == cat_val)
                    & (~all_products[product_id_col].astype(str).isin(pos_ids))
                ]
                if len(cat_candidates) >= self.num_negatives // 2:
                    sampled = cat_candidates.sample(
                        n=self.num_negatives // 2, random_state=42
                    )
                    for _, cand in sampled.iterrows():
                        nr = row.copy()
                        nr[product_col] = cand[product_col]
                        nr[product_id_col] = cand[product_id_col]
                        nr[label_col] = 0
                        nr["negative_type"] = "same_category"
                        neg_rows.append(nr)
            # Rastgele farklı kategori
            other = all_products[~all_products[product_id_col].astype(str).isin(pos_ids)]
            if len(other) > 0:
                n_rand = min(self.num_negatives - len(neg_rows), len(other))
                sampled = other.sample(n=n_rand, random_state=42)
                for _, cand in sampled.iterrows():
                    nr = row.copy()
                    nr[product_col] = cand[product_col]
                    nr[product_id_col] = cand[product_id_col]
                    nr[label_col] = 0
                    nr["negative_type"] = "random"
                    neg_rows.append(nr)

        neg_df = pd.DataFrame(neg_rows)
        return pd.concat([pos_df, neg_df], ignore_index=True)
