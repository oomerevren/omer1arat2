"""
Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 entegrasyonu.

Bu model:
- Trendyol e-ticaret veri seti üzerinde fine-tune edilmiş
- 768-dim multilingual embedding
- 384 token max sequence
- Matryoshka loss (768, 512, 128 boyutları)

Kullanım:
    embedder = TrendyolEmbedder()
    df = embedder.cosine_features(df)
    df = embedder.matryoshka_features(df, q_col="search_query", t_col="product_name")
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    SentenceTransformer = None


class TrendyolEmbedder:
    """Trendyol açık embedding modeli üzerinden feature üretimi."""

    MODEL_NAME = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"

    def __init__(
        self,
        truncate_dim: int = 768,
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        if not HAS_TORCH or not HAS_ST:
            raise ImportError(
                "TrendyolEmbedder requires torch and sentence-transformers. "
                "Install: pip install torch sentence-transformers"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(
            self.MODEL_NAME,
            trust_remote_code=True,
            truncate_dim=truncate_dim,
            device=self.device,
        )
        self.model.eval()
        # Matryoshka boyutları için cache (tekrar indirme engelle)
        self._matryoshka_cache: dict[int, SentenceTransformer] = {}

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def encode(self, texts: List[str]) -> np.ndarray:
        """Liste metinleri truncate_dim boyutlu embedding'e dönüştürür."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    def cosine_features(
        self,
        df: pd.DataFrame,
        q_col: str = "search_query",
        t_col: str = "product_name",
        prefix: str = "trendyol",
    ) -> pd.DataFrame:
        """
        DataFrame'e 4 yeni feature ekler:
        - {prefix}_embed_cosine: query-product cosine similarity
        - {prefix}_embed_dot: dot product
        - {prefix}_embed_q_norm: query embedding L2 norm
        - {prefix}_embed_p_norm: product embedding L2 norm
        """
        q_emb = self.encode(df[q_col].astype(str).tolist())
        p_emb = self.encode(df[t_col].astype(str).tolist())

        # Normalize=True ise dot product = cosine similarity
        cos = (q_emb * p_emb).sum(axis=1)
        dot = cos if self.normalize else (q_emb * p_emb).sum(axis=1)

        out = df.copy()
        out[f"{prefix}_embed_cosine"] = cos.astype(np.float32)
        out[f"{prefix}_embed_dot"] = dot.astype(np.float32)
        out[f"{prefix}_embed_q_norm"] = np.linalg.norm(q_emb, axis=1).astype(np.float32)
        out[f"{prefix}_embed_p_norm"] = np.linalg.norm(p_emb, axis=1).astype(np.float32)
        return out

    def _get_matryoshka_model(self, dim: int) -> SentenceTransformer:
        """Cache'lenmiş Matryoshka boyutlu model döndürür."""
        if dim not in self._matryoshka_cache:
            self._matryoshka_cache[dim] = SentenceTransformer(
                self.MODEL_NAME,
                trust_remote_code=True,
                truncate_dim=dim,
                device=self.device,
            )
        return self._matryoshka_cache[dim]

    def matryoshka_features(
        self,
        df: pd.DataFrame,
        q_col: str = "search_query",
        t_col: str = "product_name",
    ) -> pd.DataFrame:
        """3 farklı boyutta (128, 512, 768) cosine — model robust kontrol."""
        out = df.copy()
        for dim in [128, 512, 768]:
            mdl = self._get_matryoshka_model(dim)
            q = mdl.encode(
                out[q_col].astype(str).tolist(),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            p = mdl.encode(
                out[t_col].astype(str).tolist(),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            out[f"trendyol_cos_d{dim}"] = (q * p).sum(axis=1).astype(np.float32)
        return out
