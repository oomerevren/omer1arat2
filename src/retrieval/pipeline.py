import pandas as pd
from .hybrid import HybridRetriever

class SearchPipeline:
    def __init__(self, bm25_index, vector_index, reranker, booster, products_df: pd.DataFrame):
        self.retriever = HybridRetriever(bm25_index, vector_index)
        self.reranker = reranker
        self.booster = booster
        self.products = products_df
        # Make sure products_df has product_id as index for fast lookup if not already
        if "product_id" in self.products.columns and self.products.index.name != "product_id":
            self.products = self.products.set_index("product_id")

    def search(self, query: str, top_k: int = 50):
        # 1. Retrieve (BM25 + Vector → RRF)
        candidates = self.retriever.search(query, k=300)
        # 2. Rerank (Cross-encoder)
        if self.reranker:
            reranked = self.reranker.rerank(query, candidates, self.products, top_k=top_k*2)
        else:
            reranked = candidates[:top_k*2]
        # 3. Boost
        if self.booster:
            final_results = self.booster.boost(query, reranked, self.products)
        else:
            final_results = reranked
            
        return final_results[:top_k]
