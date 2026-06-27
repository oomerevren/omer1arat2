import pandas as pd

class BusinessBooster:
    """Apply brand, category, popularity boosts."""
    def __init__(self, boost_config: dict):
        self.boost_config = boost_config

    def _extract_brand(self, query: str):
        # Basit heuristic, tam implementasyon için NLP kullanılabilir
        return None

    def _extract_gender(self, query: str):
        if "kadın" in query or "bayan" in query: return "kadın"
        if "erkek" in query: return "erkek"
        return None

    def _extract_color(self, query: str):
        colors = ["kırmızı", "siyah", "beyaz", "mavi", "yeşil", "sarı"]
        for c in colors:
            if c in query: return c
        return None

    def boost(self, query: str, ranked_results: list, products_df: pd.DataFrame):
        query_brand = self._extract_brand(query)
        query_gender = self._extract_gender(query)
        query_color = self._extract_color(query)
        
        boosted = []
        for pid, score in ranked_results:
            # pid üzerinden ürünü bul, dict veya series olarak
            if pid not in products_df.index:
                continue
            prod = products_df.loc[pid]
            multiplier = 1.0
            
            if query_brand and prod.get("brand", "").lower() == query_brand:
                multiplier *= self.boost_config.get("brand_match", 1.3)
            if query_gender and prod.get("gender") == query_gender:
                multiplier *= self.boost_config.get("gender_match", 1.15)
            if query_color and prod.get("product_color", "").lower() == query_color:
                multiplier *= self.boost_config.get("color_match", 1.1)
                
            # Popularity (CTR, sales)
            popularity = prod.get("popularity_score", 0.5)
            multiplier *= (1 + 0.1 * float(popularity))
            
            # Stock availability
            if prod.get("in_stock", True) is False:
                multiplier *= 0.0  # filter out
                
            boosted.append((pid, score * multiplier))
            
        return sorted(boosted, key=lambda x: x[1], reverse=True)
