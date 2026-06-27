import pandas as pd

class CrossEncoderReranker:
    def __init__(self, model_path: str, batch_size: int = 32):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_path)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list, products_df: pd.DataFrame, top_k: int = 50):
        # candidates: [(product_id, score), ...]
        cand_ids = [c[0] for c in candidates]
        cand_products = products_df.set_index("product_id").loc[cand_ids]
        pairs = [(query, prod) for prod in cand_products["product_name"]]
        
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
