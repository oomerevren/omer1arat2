"""
Stratified / Group KFold doğrulama.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import hashlib
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedGroupKFold


@dataclass
class FoldResult:
    fold: int
    macro_f1: float
    precision: float
    recall: float
    train_size: int
    val_size: int


@dataclass
class ValidationReport:
    method: str
    n_splits: int
    fold_results: List[FoldResult] = field(default_factory=list)
    overall_mean_f1: float = 0.0
    overall_std_f1: float = 0.0
    stability_score: float = 0.0
    confusion: Optional[List[List[int]]] = None
    # Aşama 11 — seed stability + leak check
    seed_stability: float = 0.0
    leak_check_passed: bool = True
    temporal_split_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrossValidator:
    """
    Aşama 11 — Bulletproof Validation Şeması.
    Destekler: StratifiedGroupKFold (leak önleme), zamansal split, multi-seed stability.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        val_cfg = config.get("validation", {})
        self.metric = val_cfg.get("metric", "macro_f1")
        self.target_std = val_cfg.get("target_std_dev", 0.005)

        sk_cfg = val_cfg.get("methods", {}).get("stratified_kfold", {})
        gk_cfg = val_cfg.get("methods", {}).get("group_kfold", {})
        sgk_cfg = val_cfg.get("methods", {}).get("stratified_group_kfold", {})

        self.use_stratified = sk_cfg.get("enabled", True)
        self.use_group = gk_cfg.get("enabled", False)
        self.use_stratified_group = sgk_cfg.get("enabled", True)  # Aşama 11: varsayılan açık
        self.n_splits = sk_cfg.get("n_splits", config.get("training", {}).get("n_folds", 5))
        self.seeds = config.get("experiment", {}).get("seeds", [42])

    def _get_group_col(self, df: pd.DataFrame) -> Optional[str]:
        """Grup kolonunu bul: önce query column hash, sonra id."""
        query_col = self.config.get("data", {}).get("query_column", "search_query")
        if query_col in df.columns:
            return query_col
        id_col = self.config.get("data", {}).get("id_column", "product_id")
        if id_col in df.columns:
            return id_col
        return None

    def _get_splits(self, df: pd.DataFrame):
        label_col = self.config.get("data", {}).get("label_column", "is_relevant")
        group_col = self._get_group_col(df)
        y = df[label_col].astype(int).values

        # Zamansal split (eğer timestamp varsa)
        ts_col = self.config.get("data", {}).get("timestamp_column", None)
        if ts_col and ts_col in df.columns:
            # Zamansal split: son %20 validation
            df_sorted = df.sort_values(ts_col)
            cutoff = int(len(df_sorted) * 0.8)
            train_idx = df_sorted.index[:cutoff].tolist()
            val_idx = df_sorted.index[cutoff:].tolist()
            return [(train_idx, val_idx)]

        # StratifiedGroupKFold — leak önleme (aynı sorgu hem train hem val'de olmaz)
        if self.use_stratified_group and group_col in df.columns:
            # Grup = query hash (aynı sorgu = aynı grup)
            groups = df[group_col].apply(
                lambda s: hashlib.md5(str(s).encode()).hexdigest()[:8]
            ).astype(str).values
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=min(self.n_splits, len(np.unique(groups))),
                    shuffle=True,
                    random_state=self.config.get("experiment", {}).get("seed", 42),
                )
                return splitter.split(df, y, groups)
            except Exception:
                pass  # Fallback

        if self.use_group and group_col in df.columns:
            groups = df[group_col].astype(str).values
            splitter = GroupKFold(n_splits=min(self.n_splits, len(np.unique(groups))))
            return splitter.split(df, y, groups)

        splitter = StratifiedKFold(
            n_splits=min(self.n_splits, len(np.unique(y))),
            shuffle=True,
            random_state=self.config.get("experiment", {}).get("seed", 42),
        )
        return splitter.split(df, y)

    def validate(
        self,
        df: pd.DataFrame,
        train_fn: Callable[[pd.DataFrame, pd.DataFrame, Dict], Any],
        eval_fn: Callable[[Any, pd.DataFrame, Dict], Dict[str, float]],
    ) -> ValidationReport:
        fold_results = []
        all_preds, all_true = [], []

        for fold_idx, (train_idx, val_idx) in enumerate(self._get_splits(df)):
            train_fold = df.iloc[train_idx].reset_index(drop=True)
            val_fold = df.iloc[val_idx].reset_index(drop=True)

            # Leak kontrolü: StratifiedGroupKFold ile aynı grup hem train hem val'de olmamalı
            leak_check = True
            group_col = self._get_group_col(df)
            if group_col and group_col in train_fold.columns and group_col in val_fold.columns:
                train_groups = set(train_fold[group_col].astype(str))
                val_groups = set(val_fold[group_col].astype(str))
                overlap = train_groups & val_groups
                if overlap:
                    leak_check = False
                    print(f"[CrossValidator] UYARI: Fold {fold_idx} leak! {len(overlap)} grup kesişim.")

            model = train_fn(train_fold, val_fold, self.config)
            try:
                metrics = eval_fn(model, val_fold, self.config)
            except Exception as e:
                print(f"[CrossValidator] Fold {fold_idx} eval hatası: {e}")
                metrics = {"macro_f1": 0.0, "precision": 0.0, "recall": 0.0}

            fold_results.append(FoldResult(
                fold=fold_idx,
                macro_f1=metrics.get("macro_f1", 0.0),
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                train_size=len(train_fold),
                val_size=len(val_fold),
            ))

            label_col = self.config.get("data", {}).get("label_column", "is_relevant")
            if hasattr(model, "predict") and label_col in val_fold.columns:
                try:
                    preds = model.predict(val_fold, self.config)
                    all_preds.extend(preds.tolist())
                    all_true.extend(val_fold[label_col].astype(int).tolist())
                except Exception as e:
                    print(f"[CrossValidator] Fold {fold_idx} predict hatası: {e}")

        f1_scores = [f.macro_f1 for f in fold_results]
        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        std_f1 = float(np.std(f1_scores)) if len(f1_scores) > 1 else 0.0
        stability = max(0.0, 1.0 - std_f1 / max(self.target_std, 1e-6))

        conf = confusion_matrix(all_true, all_preds).tolist() if all_preds else None

        return ValidationReport(
            method="stratified_group_kfold" if self.use_stratified_group else ("group_kfold" if self.use_group else "stratified_kfold"),
            n_splits=len(fold_results),
            fold_results=fold_results,
            overall_mean_f1=mean_f1,
            overall_std_f1=std_f1,
            stability_score=stability,
            confusion=conf,
            leak_check_passed=leak_check,
        )

    def validate_multi_seed(
        self,
        df: pd.DataFrame,
        train_fn: Callable[[pd.DataFrame, pd.DataFrame, Dict], Any],
        eval_fn: Callable[[Any, pd.DataFrame, Dict], Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Aşama 11 — Multi-seed stability check.
        Farklı seed'ler ile CV koş, std < 0.005 ise validation güvenilir.
        """
        all_seed_scores = []
        for seed in self.seeds:
            self.config.setdefault("experiment", {})["seed"] = seed
            report = self.validate(df, train_fn, eval_fn)
            all_seed_scores.append(report.overall_mean_f1)
            print(f"[Seed {seed}] CV Mean F1: {report.overall_mean_f1:.4f}")

        seed_std = float(np.std(all_seed_scores)) if len(all_seed_scores) > 1 else 0.0
        seed_mean = float(np.mean(all_seed_scores)) if all_seed_scores else 0.0
        seed_stability = max(0.0, 1.0 - seed_std / max(self.target_std, 1e-6))
        reliable = seed_std < self.target_std

        print(f"\n[Seed Stability] Mean: {seed_mean:.4f} ± {seed_std:.4f} (std < {self.target_std}) → Reliable: {reliable}")
        return {
            "seed_scores": all_seed_scores,
            "seed_mean": seed_mean,
            "seed_std": seed_std,
            "seed_stability": seed_stability,
            "reliable": reliable,
        }
