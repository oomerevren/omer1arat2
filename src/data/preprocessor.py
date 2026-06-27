import re
import string
import warnings
from pathlib import Path

# Optional libraries for advanced usage
try:
    import symspellpy
    HAS_SYMSPELL = True
except ImportError:
    HAS_SYMSPELL = False

try:
    import zeyrek  # Zemberek alternative for Python
    HAS_ZEYREK = True
except ImportError:
    HAS_ZEYREK = False


class TextPreprocessor:
    """
    Turkish E-commerce Text Preprocessor.
    Handles lowercase conversion (I/ı, İ/i), punctuation, typo correction, and lemmatization.
    """

    @classmethod
    def from_config(cls, config) -> "TextPreprocessor":
        """YAML config veya dict'ten preprocessor oluşturur."""
        if isinstance(config, dict):
            prep = config.get("data", {}).get("preprocessing", {})
            lem = prep.get("lemmatizer", "zemberek")
            return cls(
                remove_punctuation=prep.get("remove_punctuation", True),
                use_typo_correction=prep.get("typo_simulation", False),
                use_lemmatization=lem not in ("none", None, False),
            )
        prep = getattr(config, "data", {})
        if hasattr(prep, "preprocessing"):
            prep = prep.preprocessing
        return cls()
    
    def __init__(
        self,
        config=None,
        remove_punctuation: bool = True,
        use_typo_correction: bool = True,
        use_lemmatization: bool = True,
    ):
        if isinstance(config, dict):
            inst = self.from_config(config)
            self.remove_punctuation = inst.remove_punctuation
            self.use_typo_correction = inst.use_typo_correction
            self.use_lemmatization = inst.use_lemmatization
        else:
            self.remove_punctuation = remove_punctuation
            self.use_typo_correction = use_typo_correction and HAS_SYMSPELL
            self.use_lemmatization = use_lemmatization and HAS_ZEYREK
        
        if use_typo_correction and not HAS_SYMSPELL:
            warnings.warn("SymSpell not installed. Typo correction is disabled. Install symspellpy.")
            
        if use_lemmatization and not HAS_ZEYREK:
            warnings.warn("Zeyrek not installed. Lemmatization is disabled. Install zeyrek.")
            
        self._init_spell_checker()
        self._init_lemmatizer()
        
    def _init_spell_checker(self):
        if self.use_typo_correction:
            from symspellpy import SymSpell
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dict_path = Path(__file__).resolve().parents[2] / "data" / "ecommerce_dict.txt"
            if dict_path.exists():
                self.sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
            
    def _init_lemmatizer(self):
        self._lemmatize_warned = False
        if self.use_lemmatization:
            import zeyrek
            self.analyzer = zeyrek.MorphAnalyzer()
            
    def _turkish_lower(self, text: str) -> str:
        """Handles Turkish character edge cases for lowercase."""
        text = text.replace('İ', 'i').replace('I', 'ı')
        return text.lower()
        
    def _clean_punctuation(self, text: str) -> str:
        """Removes punctuation using regex."""
        return re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
    def _correct_typos(self, text: str) -> str:
        """Corrects typos using SymSpell."""
        if not self.use_typo_correction:
            return text
        
        words = text.split()
        corrected_words = []
        for word in words:
            suggestions = self.sym_spell.lookup(
                word, 
                verbosity=symspellpy.Verbosity.CLOSEST, 
                max_edit_distance=2
            )
            if suggestions:
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)
        
    def _lemmatize(self, text: str) -> str:
        """Lemmatizes Turkish text using Zeyrek (Zemberek logic)."""
        if not self.use_lemmatization:
            return text
            
        try:
            results = self.analyzer.analyze(text)
            lemmatized = []
            for res in results:
                if res and res[0].lemmas:
                    lemmatized.append(res[0].lemmas[0])
                else:
                    lemmatized.append(res[0].word)
            return " ".join(lemmatized)
        except Exception as e:
            if not self._lemmatize_warned:
                warnings.warn(f"Lemmatization failed, falling back to raw text: {e}")
                self._lemmatize_warned = True
            return text
            
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        if not isinstance(text, str):
            return ""
            
        text = str(text).strip()
        text = self._turkish_lower(text)
        
        if self.remove_punctuation:
            text = self._clean_punctuation(text)
            
        # Standardize whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        text = self._correct_typos(text)
        text = self._lemmatize(text)
        
        return text

# Factory method to match tests
def ZemberekLemmatizer():
    return TextPreprocessor(remove_punctuation=False, use_typo_correction=False, use_lemmatization=True)
