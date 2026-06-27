class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) of BM25 + Vector."""
    def __init__(self, bm25_index, vector_index, rrf_k: int = 60):
        self.bm25 = bm25_index
        self.vector = vector_index
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 300):
        bm25_results = self.bm25.search(query, k=k)
        vector_results = self.vector.search(query, k=k)
        
        # RRF fusion
        scores = {}
        for rank, (pid, _) in enumerate(bm25_results):
            scores[pid] = scores.get(pid, 0) + 1.0 / (self.rrf_k + rank)
        for rank, (pid, _) in enumerate(vector_results):
            scores[pid] = scores.get(pid, 0) + 1.0 / (self.rrf_k + rank)
            
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
