import numpy as np
import pandas as pd
import faiss

class FAISSIndex:
    def __init__(self, embedder, dim: int = 768):
        self.embedder = embedder
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized)
        self.product_ids = []

    def build(self, products: pd.DataFrame, text_col: str = "product_name", batch_size: int = 256):
        self.product_ids = products["product_id"].tolist()
        embeddings = self.embedder.encode(products[text_col].astype(str).tolist())
        # Normalize (cosine = inner product on normalized vectors)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, k: int = 200):
        q_emb = self.embedder.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), k)
        return [(self.product_ids[indices[0][i]], scores[0][i]) for i in range(min(k, len(self.product_ids)))]
