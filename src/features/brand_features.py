import pandas as pd
import warnings

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

class BrandFeatureExtractor:
    """
    Extracts Fuzzy Brand matching features using RapidFuzz.
    """
    
    def __init__(self, brand_col: str = "brand", query_col: str = "search_query"):
        self.brand_col = brand_col
        self.query_col = query_col
        
        if not HAS_RAPIDFUZZ:
            warnings.warn("RapidFuzz not installed. Falling back to basic overlap.")

    def _fuzzy_match(self, query: str, brand: str) -> float:
        if not isinstance(query, str) or not isinstance(brand, str) or not brand:
            return 0.0
            
        query = query.lower()
        brand = brand.lower()
        
        if not HAS_RAPIDFUZZ:
            return 100.0 if brand in query else 0.0
            
        # Partial ratio gives high score if brand is a substring of query
        return fuzz.partial_ratio(brand, query)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.query_col not in df.columns or self.brand_col not in df.columns:
            return df
            
        df = df.copy()
        
        df["fuzzy_brand_match_score"] = df.apply(
            lambda row: self._fuzzy_match(row[self.query_col], row.get(self.brand_col, "")), 
            axis=1
        )
        
        # Binary indicator if score > 85
        df["has_brand_match"] = (df["fuzzy_brand_match_score"] > 85).astype(int)
        
        return df

# Alias for tests
FuzzyBrandMatcher = BrandFeatureExtractor
