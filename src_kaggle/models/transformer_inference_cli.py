from __future__ import annotations
from pathlib import Path
import pandas as pd
from src_kaggle.models.transformer_trainer import TransformerCrossEncoder

def predict_fold_checkpoint(df: pd.DataFrame, checkpoint_dir: str, config: dict) -> pd.DataFrame:
    model = TransformerCrossEncoder.load(checkpoint_dir, config)
    proba = model.predict_proba(df)
    return pd.DataFrame({'proba': proba})
