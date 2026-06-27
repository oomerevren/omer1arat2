import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.product_ids = []
        self.product_texts = []

    def build(self, products: pd.DataFrame, text_col: str = "product_name"):
        self.product_ids = products["product_id"].tolist()
        self.product_texts = products[text_col].astype(str).tolist()
        tokenized = [t.split() for t in self.product_texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 200):
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.product_ids[i], scores[i]) for i in top_idx]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "ids": self.product_ids}, f)
