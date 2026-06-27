import random
import pandas as pd
from typing import List, Optional, Set, Dict

class DataAugmenter:
    """
    Data Augmentation for E-commerce Search Queries.
    Implements Synonym Replacement and Word Dropout to make the model robust against user variations.
    """
    
    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None, dropout_prob: float = 0.1):
        self.synonym_dict = synonym_dict or {
            "telefon": ["cep telefonu", "smartphone", "mobil"],
            "kırmızı": ["bordo", "nar çiçeği"],
            "ayakkabı": ["sneaker", "pabuç", "spor ayakkabı"],
            "bilgisayar": ["pc", "laptop", "dizüstü"],
        }
        self.dropout_prob = dropout_prob

    def synonym_replacement(self, text: str) -> str:
        """Replaces words with their synonyms with a certain probability."""
        words = text.split()
        new_words = []
        for word in words:
            if word in self.synonym_dict and random.random() < 0.5:
                new_words.append(random.choice(self.synonym_dict[word]))
            else:
                new_words.append(word)
        return " ".join(new_words)

    def word_dropout(self, text: str) -> str:
        """Randomly drops words from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = [w for w in words if random.random() > self.dropout_prob]
        
        # Ensure we don't drop everything
        if not new_words:
            new_words = [random.choice(words)]
            
        return " ".join(new_words)

    def augment(self, text: str, n_variants: int = 1) -> List[str]:
        """Generate n augmented variants of the input text."""
        variants: Set[str] = set()
        attempts = 0
        while len(variants) < n_variants and attempts < n_variants * 3:
            attempts += 1
            variant = self.word_dropout(self.synonym_replacement(text))
            if variant != text:
                variants.add(variant)
                
        return list(variants)

    def augment_dataframe(self, df: pd.DataFrame, text_col: str, n_variants: int = 1) -> pd.DataFrame:
        """Augments a DataFrame by duplicating rows with augmented text."""
        augmented_rows = []
        
        for _, row in df.iterrows():
            variants = self.augment(row[text_col], n_variants)
            for var in variants:
                new_row = row.copy()
                new_row[text_col] = var
                augmented_rows.append(new_row)
                
        if augmented_rows:
            aug_df = pd.DataFrame(augmented_rows)
            return pd.concat([df, aug_df], ignore_index=True)
        return df
