import pandas as pd
import warnings
from typing import List, Dict

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

class BM25Extractor:
    """
    Extracts lexical features between query and product fields.
    Includes BM25 score, Jaccard Similarity, and Longest Common Substring.
    """
    
    def __init__(self, text_col: str = "product_name", query_col: str = "search_query"):
        self.text_col = text_col
        self.query_col = query_col
        self.bm25_model = None
        
        if not HAS_BM25:
            warnings.warn("rank_bm25 not installed. BM25 score will be 0. Install rank-bm25.")

    def fit(self, df: pd.DataFrame):
        """Fits the BM25 index on the corpus."""
        if not HAS_BM25 or self.text_col not in df.columns:
            return
            
        corpus = df[self.text_col].fillna("").astype(str).tolist()
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        
    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        set1 = set(s1.split())
        set2 = set(s2.split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def _longest_common_substring_len(self, s1: str, s2: str) -> int:
        if not s1 or not s2:
            return 0
        m = len(s1)
        n = len(s2)
        lcs = 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    lcs = max(lcs, dp[i][j])
                else:
                    dp[i][j] = 0
        return lcs

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.query_col not in df.columns or self.text_col not in df.columns:
            return df
            
        df = df.copy()
        queries = df[self.query_col].fillna("").astype(str)
        texts = df[self.text_col].fillna("").astype(str)
        
        # Jaccard
        df["jaccard_similarity"] = [self._jaccard_similarity(q, t) for q, t in zip(queries, texts)]
        
        # LCS
        df["longest_common_substring"] = [self._longest_common_substring_len(q, t) for q, t in zip(queries, texts)]
        
        # BM25 — satır bazlı skor (query vs product text)
        bm25_scores = []
        for q, t in zip(queries, texts):
            if not q.strip() or not t.strip():
                bm25_scores.append(0.0)
                continue
            if HAS_BM25:
                from rank_bm25 import BM25Okapi
                mini = BM25Okapi([t.split()])
                bm25_scores.append(float(mini.get_scores(q.split())[0]))
            else:
                bm25_scores.append(0.0)

        df["bm25_score"] = bm25_scores
        df["bm25_approx_score"] = df["bm25_score"]
        
        return df
