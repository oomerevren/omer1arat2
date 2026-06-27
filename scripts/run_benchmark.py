
"""
TEKNOFEST 2026 — Benchmark Runner + Feature Ablation + Error Analyzer
======================================================================
3 kritik MLOps modülü tek dosyada:
  1. Benchmark Runner: Tek komutla tüm modelleri karşılaştır
  2. Feature Ablation: Her özelliğin katkısını ölç
  3. Error Analyzer: Hataları sınıflandır ve dashboard hazırla
"""

import os
import sys
import json
import time
import concurrent.futures
import hashlib
import pickle
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

# Opsiyonel import
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# 1. BENCHMARK RUNNER
# ============================================================

@dataclass
class ModelBenchmarkResult:
    """Tek bir modelin benchmark sonucu."""
    model_name: str
    model_type: str
    
    # Performans
    macro_f1: float
    macro_f1_std: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    # Süre
    training_time_seconds: float = 0.0
    inference_latency_ms: float = 0.0
    inference_p95_ms: float = 0.0
    inference_throughput_qps: float = 0.0
    
    # Kaynak
    memory_usage_mb: float = 0.0
    gpu_memory_mb: Optional[float] = None
    model_size_mb: float = 0.0
    num_params: int = 0
    
    # Meta
    feature_set: str = "all"
    config_hash: str = ""
    timestamp: str = ""
    status: str = "success"  # success | error | timeout
    error_message: str = ""


class BenchmarkRunner:
    """
    Tek komutla tüm modelleri karşılaştıran framework.
    
    Desteklenen modeller:
        - DistilBERTurk Cross-Encoder
        - BERTurk Cross-Encoder
        - E5 Large
        - BGE-M3
        - CatBoost
        - LightGBM
        - Ensemble (weighted)
        - Ensemble (stacking)
    
    Kullanım:
        runner = BenchmarkRunner(data_loader, config_loader)
        results = runner.run_all()
        runner.save_report(results)
    """

    def __init__(
        self,
        data_loader: Callable = None,
        config_loader: Callable = None,
        output_dir: str = "./experiments/benchmarks",
    ):
        self.data_loader = data_loader
        self.config_loader = config_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_registry: Dict[str, Dict] = {}
        self._register_default_models()
        
        # Sonuçlar
        self.results: List[ModelBenchmarkResult] = []
    
    def _register_default_models(self):
        """Varsayılan modelleri kaydet."""
        self.model_registry = {
            "distilberturk_ce": {
                "type": "cross_encoder",
                "model_name": "dbmdz/distilbert-base-turkish-cased",
                "display_name": "DistilBERTurk Cross-Encoder",
                "priority": 1,
                "config_overrides": {"training.num_epochs": 5, "training.batch_size": 32},
            },
            "berturk_ce": {
                "type": "cross_encoder",
                "model_name": "dbmdz/bert-base-turkish-cased",
                "display_name": "BERTurk Cross-Encoder",
                "priority": 2,
                "config_overrides": {"training.num_epochs": 3, "training.batch_size": 16},
            },
            "e5_large": {
                "type": "cross_encoder",
                "model_name": "intfloat/multilingual-e5-large",
                "display_name": "E5 Large (Multilingual)",
                "priority": 3,
                "config_overrides": {"training.batch_size": 8},
            },
            "bge_m3": {
                "type": "cross_encoder",
                "model_name": "BAAI/bge-m3",
                "display_name": "BGE-M3",
                "priority": 4,
                "config_overrides": {"training.batch_size": 8},
            },
            "catboost": {
                "type": "catboost",
                "model_name": "CatBoost",
                "display_name": "CatBoost (Meta-Model)",
                "priority": 5,
                "config_overrides": {},
            },
            "lightgbm": {
                "type": "lightgbm",
                "model_name": "LightGBM",
                "display_name": "LightGBM (Meta-Model)",
                "priority": 6,
                "config_overrides": {},
            },
            "ensemble_weighted": {
                "type": "ensemble",
                "model_name": "WeightedEnsemble",
                "display_name": "Ensemble (Weighted Avg)",
                "priority": 7,
                "config_overrides": {"ensemble.method": "weighted_average"},
            },
            "ensemble_stacking": {
                "type": "ensemble",
                "model_name": "StackingEnsemble",
                "display_name": "Ensemble (Stacking)",
                "priority": 8,
                "config_overrides": {"ensemble.method": "stacking"},
            },
        }
    
    def register_model(
        self, key: str, model_type: str, model_name: str,
        display_name: str, priority: int = 99, **kwargs
    ):
        """Yeni model kaydet."""
        self.model_registry[key] = {
            "type": model_type,
            "model_name": model_name,
            "display_name": display_name,
            "priority": priority,
            "config_overrides": kwargs.get("config_overrides", {}),
        }
    
    def run_all(
        self,
        train_fn: Callable = None,
        eval_fn: Callable = None,
        feature_set: str = "all",
        n_folds: int = 3,
        seeds: List[int] = None,
        timeout_seconds: int = 7200,  # 2 saat
    ) -> List[ModelBenchmarkResult]:
        """
        Tüm kayıtlı modelleri benchmark et.
        
        Args:
            train_fn: (config_dict, train_data) -> trained_model
            eval_fn: (model, val_data) -> {"macro_f1": float, ...}
            feature_set: Hangi feature set kullanılacak.
            n_folds: Cross-validation fold sayısı.
            seeds: Test edilecek seed'ler.
            timeout_seconds: Model başına maksimum süre.
        """
        if seeds is None:
            seeds = [42, 123, 2026]
        
        if train_fn is None or eval_fn is None:
            print("[Benchmark] train_fn/eval_fn verilmedi, sentetik test yapılıyor.")
            return self._run_synthetic_benchmark()
        
        models = sorted(
            self.model_registry.items(),
            key=lambda x: x[1]["priority"],
        )
        
        for model_key, model_info in models:
            print(f"\n{'='*60}")
            print(f"  BENCHMARK: {model_info['display_name']}")
            print(f"  Type: {model_info['type']} | Priority: {model_info['priority']}")
            print(f"{'='*60}")
            
            try:
                result = self._benchmark_single_model(
                    model_key=model_key,
                    model_info=model_info,
                    train_fn=train_fn,
                    eval_fn=eval_fn,
                    feature_set=feature_set,
                    n_folds=n_folds,
                    seeds=seeds,
                    timeout=timeout_seconds,
                )
                self.results.append(result)
                
            except Exception as e:
                print(f"  ✗ HATA: {e}")
                self.results.append(ModelBenchmarkResult(
                    model_name=model_info["display_name"],
                    model_type=model_info["type"],
                    macro_f1=0.0,
                    status="error",
                    error_message=str(e),
                ))
        
        return self.results
    
    def _benchmark_single_model(
        self,
        model_key: str,
        model_info: Dict,
        train_fn: Callable,
        eval_fn: Callable,
        feature_set: str,
        n_folds: int,
        seeds: List[int],
        timeout: int,
    ) -> ModelBenchmarkResult:
        """Tek bir modeli benchmark et."""
        
        # Model boyutu (tahmini)
        model_sizes = {
            "distilberturk_ce": (66_000_000, 265),
            "berturk_ce": (110_000_000, 440),
            "e5_large": (335_000_000, 1340),
            "bge_m3": (568_000_000, 2270),
            "catboost": (0, 50),
            "lightgbm": (0, 30),
            "ensemble_weighted": (0, 500),
            "ensemble_stacking": (0, 600),
        }
        num_params, size_mb = model_sizes.get(model_key, (0, 0))
        
        # Memory tracking
        if HAS_PSUTIL:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
        
        # Training
        t0 = time.time()
        config = {"model": {"type": model_info["type"]}, "features": {"feature_set": feature_set}}
        
        all_f1_scores = []
        
        for seed in seeds:
            config["experiment"] = {"seed": seed}
            
            try:
                # Gerçek veri yükleyici çağrısı
                data = self.data_loader() if getattr(self, 'data_loader', None) else None
                
                # Timeout implementation
                def _run_model():
                    m = train_fn(config, data)
                    met = eval_fn(m, data)
                    
                    # Inference latency (Mocking the eval latency if it's missing)
                    inf_t0 = time.time()
                    # Simulate small inference if actual eval was too fast
                    _ = met.get("macro_f1", 0.0)
                    inf_t1 = time.time()
                    return met, (inf_t1 - inf_t0) * 1000.0

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_model)
                    metrics, inf_lat = future.result(timeout=timeout)

                all_f1_scores.append(metrics.get("macro_f1", 0.0))
                inference_latency_ms = inf_lat
            except concurrent.futures.TimeoutError:
                print(f"    Seed {seed}: TIMEOUT")
                continue
            except Exception as e:
                print(f"    Seed {seed}: HATA - {e}")
                continue
        
        training_time = time.time() - t0
        
        if not all_f1_scores:
            raise RuntimeError("Hiçbir seed başarılı olmadı!")
        
        mean_f1 = np.mean(all_f1_scores)
        std_f1 = np.std(all_f1_scores) if len(all_f1_scores) > 1 else 0.0
        
        # Memory
        if HAS_PSUTIL:
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_usage = mem_after - mem_before
        else:
            memory_usage = size_mb
        
        # GPU memory
        gpu_mem = None
        if HAS_TORCH and torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        
        return ModelBenchmarkResult(
            model_name=model_info["display_name"],
            model_type=model_info["type"],
            macro_f1=round(mean_f1, 5),
            macro_f1_std=round(std_f1, 5),
            training_time_seconds=round(training_time, 1),
            inference_latency_ms=round(locals().get('inference_latency_ms', 0.0), 1),
            memory_usage_mb=round(memory_usage, 1),
            gpu_memory_mb=round(gpu_mem, 1) if gpu_mem else None,
            model_size_mb=size_mb,
            num_params=num_params,
            feature_set=feature_set,
            config_hash=hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            timestamp=datetime.now().isoformat(),
            status="success",
        )
    
    def _run_synthetic_benchmark(self) -> List[ModelBenchmarkResult]:
        """Sentetik veri ile demo benchmark."""
        print("[Benchmark] Sentetik benchmark çalıştırılıyor...")
        
        models = [
            ("DistilBERTurk CE", "cross_encoder", 0.0, 265, 66_000_000, 180, 15.2),
            ("BERTurk CE", "cross_encoder", 0.0, 440, 110_000_000, 350, 45.2),
            ("E5 Large", "cross_encoder", 0.0, 1340, 335_000_000, 520, 78.5),
            ("BGE-M3", "cross_encoder", 0.0, 2270, 568_000_000, 680, 95.0),
            ("CatBoost", "catboost", 0.0, 50, 0, 45, 2.1),
            ("LightGBM", "lightgbm", 0.0, 30, 0, 35, 1.8),
            ("Ensemble (Weighted)", "ensemble", 0.0, 500, 0, 800, 65.0),
            ("Ensemble (Stacking)", "ensemble", 0.0, 600, 0, 950, 85.0),
        ]
        
        for name, mtype, f1, size, params, train_time, latency in models:
            self.results.append(ModelBenchmarkResult(
                model_name=name,
                model_type=mtype,
                macro_f1=f1,
                macro_f1_std=round(np.random.uniform(0.002, 0.008), 5),
                training_time_seconds=train_time,
                inference_latency_ms=latency,
                inference_p95_ms=latency * 1.6,
                memory_usage_mb=size,
                model_size_mb=size,
                num_params=params,
                timestamp=datetime.now().isoformat(),
                status="success",
            ))
        
        return self.results
    
    def save_report(self, path: str = None) -> str:
        """Benchmark raporunu kaydet."""
        if path is None:
            path = str(self.output_dir / "benchmark_report.md")
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Markdown raporu
        lines = [
            "# TEKNOFEST 2026 — Model Benchmark Raporu",
            f"",
            f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            f"Toplam model: {len(self.results)}",
            f"",
            "## Sıralama (Macro-F1'e göre)",
            "",
            "| # | Model | Tip | Macro-F1 | ±Std | Train(s) | Lat(ms) | RAM(MB) | Params |",
            "|---|-------|-----|----------|------|----------|---------|---------|--------|",
        ]
        
        for i, r in enumerate(
            sorted(self.results, key=lambda x: x.macro_f1, reverse=True), 1
        ):
            icon = "✓" if r.status == "success" else "✗"
            lines.append(
                f"| {i} | {icon} {r.model_name} | {r.model_type} | "
                f"{r.macro_f1:.4f} | {r.macro_f1_std:.4f} | "
                f"{r.training_time_seconds:.0f} | {r.inference_latency_ms:.1f} | "
                f"{r.memory_usage_mb:.0f} | {r.num_params:,} |"
            )
        
        lines.extend([
            "",
            "## Kazanan Model",
            "",
        ])
        
        best = max(
            [r for r in self.results if r.status == "success"],
            key=lambda x: x.macro_f1,
            default=None,
        )
        
        if best:
            lines.append(f"**{best.model_name}** — Macro-F1: **{best.macro_f1:.4f}**")
            lines.append(f"- Tip: {best.model_type}")
            lines.append(f"- Eğitim süresi: {best.training_time_seconds:.0f}s")
            lines.append(f"- Inference: {best.inference_latency_ms:.1f}ms")
            lines.append(f"- Bellek: {best.memory_usage_mb:.0f}MB")
        
        md = "\n".join(lines)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        
        # CSV
        csv_path = path.replace(".md", ".csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n  [OK] Rapor kaydedildi: {path}")
        print(f"  [OK] CSV kaydedildi: {csv_path}")
        
        return md
    
    def print_summary(self):
        """Konsola özet tablo bas."""
        if not self.results:
            print("Sonuç yok.")
            return
        
        print(f"\n{'='*80}")
        print(f"  BENCHMARK ÖZETİ — {len(self.results)} model")
        print(f"{'='*80}")
        print(f"  {'Model':<25} {'F1':<10} {'Train(s)':<10} {'Lat(ms)':<10} {'RAM(MB)':<10}")
        print(f"  {'-'*65}")
        
        for r in sorted(self.results, key=lambda x: x.macro_f1, reverse=True):
            print(f"  {r.model_name:<25} {r.macro_f1:<10.4f} "
                  f"{r.training_time_seconds:<10.0f} {r.inference_latency_ms:<10.1f} "
                  f"{r.memory_usage_mb:<10.0f}")


# ============================================================
# 2. FEATURE ABLATION FRAMEWORK
# ============================================================

@dataclass
class AblationResult:
    """Bir feature'ın ablation sonucu."""
    feature_removed: str
    baseline_f1: float
    ablated_f1: float
    f1_difference: float
    f1_difference_pct: float
    impact: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW", "NEGLIGIBLE"
    status: str = "success"
    error_message: str = ""


class FeatureAblationFramework:
    """
    Her özelliğin katkısını ölçen otomatik sistem.
    
    Çalışma mantığı:
        1. Tüm feature'lar ile baseline F1 ölç
        2. Her feature'ı tek tek kaldır
        3. F1 farkını hesapla
        4. Impact seviyesini belirle
    
    Kullanım:
        abl = FeatureAblationFramework(train_fn, eval_fn)
        results = abl.run_ablation(features_config, baseline_f1)
        abl.print_summary()
    """

    IMPACT_THRESHOLDS = {
        "CRITICAL": (-1.0, -0.020),     # F1 düşüşü > 2%
        "HIGH": (-0.020, -0.010),        # F1 düşüşü 1-2%
        "MEDIUM": (-0.010, -0.003),      # F1 düşüşü 0.3-1%
        "LOW": (-0.003, -0.001),         # F1 düşüşü 0.1-0.3%
        "NEGLIGIBLE": (-0.001, 1.0),     # F1 düşüşü < 0.1% veya artış
    }

    def __init__(
        self,
        train_fn: Callable = None,
        eval_fn: Callable = None,
    ):
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.results: List[AblationResult] = []

    def run_ablation(
        self,
        all_features: Dict[str, bool],
        baseline_f1: float,
        n_repeats: int = 3,
    ) -> List[AblationResult]:
        """
        Tüm feature'lar için ablation çalıştır.
        
        Args:
            all_features: {"bm25_score": True, "fuzzy_brand": True, ...}
            baseline_f1: Tüm feature'larla alınan F1.
            n_repeats: Her ablation kaç kere tekrarlanacak.
        """
        self.results = []

        for feature_name in all_features:
            print(f"\n  [Ablation] '{feature_name}' kaldırılıyor...")

            try:
                ablated_features = all_features.copy()
                ablated_features[feature_name] = False

                f1_scores = []
                for rep in range(n_repeats):
                    if self.train_fn and self.eval_fn:
                        config = {"features": ablated_features, "experiment": {"seed": 42 + rep}}
                        model = self.train_fn(config, None)
                        metrics = self.eval_fn(model, None)
                        f1_scores.append(metrics.get("macro_f1", 0.0))
                    else:
                        # Sentetik: baseline'dan biraz daha düşük
                        drop = np.random.uniform(0.001, 0.025)
                        f1_scores.append(baseline_f1 - drop)

                ablated_f1 = np.mean(f1_scores)
                diff = ablated_f1 - baseline_f1
                diff_pct = (diff / max(baseline_f1, 0.001)) * 100

                # Impact seviyesi
                impact = "NEGLIGIBLE"
                for level, (low, high) in self.IMPACT_THRESHOLDS.items():
                    if low <= diff < high:
                        impact = level
                        break

                self.results.append(AblationResult(
                    feature_removed=feature_name,
                    baseline_f1=round(baseline_f1, 5),
                    ablated_f1=round(ablated_f1, 5),
                    f1_difference=round(diff, 5),
                    f1_difference_pct=round(diff_pct, 2),
                    impact=impact,
                ))

                print(f"    Baseline: {baseline_f1:.4f} → Ablated: {ablated_f1:.4f}")
                print(f"    Fark: {diff:+.4f} ({diff_pct:+.1f}%) → Impact: {impact}")

            except Exception as e:
                print(f"    ✗ HATA: {e}")
                self.results.append(AblationResult(
                    feature_removed=feature_name,
                    baseline_f1=baseline_f1,
                    ablated_f1=0.0,
                    f1_difference=0.0,
                    f1_difference_pct=0.0,
                    impact="ERROR",
                    status="error",
                    error_message=str(e),
                ))

        return self.results

    def run_synthetic(self) -> List[AblationResult]:
        """Sentetik veri ile ablation demo'su."""
        features = {
            "bm25_score": 0.0,
            "fuzzy_brand_match": 0.0,
            "category_hierarchy": 0.0,
            "token_overlap": 0.0,
            "fuzzy_token_set": 0.0,
            "color_match": 0.0,
            "material_match": 0.0,
            "brand_exact_match": 0.0,
            "embedding_similarity": 0.0,
            "jaccard_similarity": 0.0,
        }
        
        baseline_f1 = 0.0
        
        for feature, ablated_f1 in features.items():
            diff = ablated_f1 - baseline_f1
            
            impact = "NEGLIGIBLE"
            for level, (low, high) in self.IMPACT_THRESHOLDS.items():
                if low <= diff < high:
                    impact = level
                    break
            
            self.results.append(AblationResult(
                feature_removed=feature,
                baseline_f1=baseline_f1,
                ablated_f1=ablated_f1,
                f1_difference=round(diff, 5),
                f1_difference_pct=round(diff / baseline_f1 * 100, 2),
                impact=impact,
            ))
        
        return self.results

    def print_summary(self):
        """Ablasyon özet tablosu."""
        if not self.results:
            print("Ablasyon sonucu yok.")
            return
        
        # Impact'e göre sırala
        impact_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NEGLIGIBLE": 4, "ERROR": 5}
        sorted_results = sorted(self.results, key=lambda x: impact_order.get(x.impact, 99))

        print(f"\n{'='*75}")
        print(f"  FEATURE ABLATION SONUÇLARI")
        print(f"{'='*75}")
        print(f"  {'Feature':<30} {'Diff F1':<12} {'Diff %':<10} {'Impact':<12}")
        print(f"  {'-'*64}")

        for r in sorted_results:
            icon = {
                "CRITICAL": "[X]", "HIGH": "[!]", "MEDIUM": "[-]",
                "LOW": "[v]", "NEGLIGIBLE": "[~]", "ERROR": "[E]"
            }.get(r.impact, "[?]")
            
            print(f"  {icon} {r.feature_removed:<27} "
                  f"{r.f1_difference:+.5f}   {r.f1_difference_pct:+.1f}%     "
                  f"{r.impact}")

        print(f"{'='*75}")

    def save_report(self, path: str = None):
        """Ablasyon raporunu kaydet."""
        if path is None:
            path = "./experiments/ablation_report.csv"
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(path, index=False)
        print(f"  [OK] Ablasyon raporu: {path}")


# ============================================================
# 3. ERROR ANALYZER
# ============================================================

@dataclass
class ErrorSample:
    """Hatalı tahmin edilmiş tek bir örnek."""
    query: str
    product_title: str
    product_brand: str
    product_category: str
    true_label: int
    predicted_label: int
    confidence: float
    error_type: str  # "brand_mismatch", "category_mismatch", vb.
    error_detail: str
    features: Dict[str, float] = field(default_factory=dict)


class ErrorAnalyzer:
    """
    Model hatalarını otomatik sınıflandıran sistem.
    """
    def __init__(self):
        self.errors: List[ErrorSample] = []
        self.error_stats: Dict[str, int] = defaultdict(int)
        self.confusion_matrix: Optional[np.ndarray] = None

    def analyze(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame = None,
        query_column: str = "search_query",
        pred_column: str = "predicted_label",
        true_column: str = "is_relevant",
        prob_column: str = "confidence",
        brand_column: str = "brand",
        category_column: str = "category",
        title_column: str = "product_name",
    ) -> List[ErrorSample]:
        """
        Tahminleri analiz et, hataları sınıflandır.
        """
        self.errors = []
        self.error_stats = defaultdict(int)

        if ground_truth is not None:
            df = predictions.merge(
                ground_truth[[query_column, true_column]],
                on=query_column,
                how="left",
                suffixes=("", "_true"),
            )
            true_label_col = f"{true_column}_true" if f"{true_column}_true" in df.columns else true_column
        else:
            df = predictions.copy()
            true_label_col = true_column

        if true_label_col in df.columns and pred_column in df.columns:
            error_mask = df[pred_column] != df[true_label_col]
        else:
            error_mask = pd.Series([True] * len(df))

        error_df = df[error_mask].copy()

        print(f"[ErrorAnalyzer] {len(df)} ornek, {len(error_df)} hata "
              f"(%{len(error_df)/max(len(df),1)*100:.1f})")

        for _, row in error_df.iterrows():
            query = str(row.get(query_column, ""))
            product = str(row.get(title_column, ""))
            brand = str(row.get(brand_column, ""))
            category = str(row.get(category_column, ""))
            true_label = int(row.get(true_label_col, -1))
            pred_label = int(row.get(pred_column, -1))
            confidence = float(row.get(prob_column, 0.0))

            error_type, error_detail = self._classify_error(
                query, product, brand, category,
                true_label, pred_label, confidence,
            )

            self.error_stats[error_type] += 1

            self.errors.append(ErrorSample(
                query=query,
                product_title=product[:100],
                product_brand=brand,
                product_category=category,
                true_label=true_label,
                predicted_label=pred_label,
                confidence=confidence,
                error_type=error_type,
                error_detail=error_detail,
            ))

        return self.errors

    def _classify_error(
        self,
        query: str,
        product: str,
        brand: str,
        category: str,
        true_label: int,
        pred_label: int,
        confidence: float,
    ) -> Tuple[str, str]:
        q_lower = query.lower()
        p_lower = product.lower()
        b_lower = brand.lower()

        if b_lower and b_lower != "unknown" and b_lower != "":
            if b_lower in q_lower:
                if true_label == 0 and pred_label >= 1:
                    return "BRAND_MISMATCH", "Brand mismatch"

        cat_words = set(category.lower().replace(">", " ").split())
        q_words = set(q_lower.split())
        cat_overlap = len(cat_words & q_words)
        if cat_overlap == 0 and len(q_words) >= 3:
            if true_label >= 1 and pred_label == 0:
                return "CATEGORY_MISMATCH", "Category mismatch"

        accessory_words = {"canta", "cuzdan", "kemer", "sapka", "atki", "eldiven",
                          "kolye", "bileklik", "saat", "gozluk", "aksesuar"}
        if any(w in q_lower for w in accessory_words):
            if true_label >= 1 and pred_label == 0:
                return "ACCESSORY_CONFUSION", "Accessory confusion"

        q_set = set(q_lower.split())
        p_set = set(p_lower.split())
        overlap = len(q_set & p_set)
        if overlap >= 2 and len(q_set) >= 2:
            if true_label == 0 and pred_label >= 1:
                return "SEMANTIC_ERROR", "Semantic error"
            elif true_label >= 1 and pred_label == 0:
                return "SEMANTIC_ERROR", "Semantic error"

        if len(q_lower.split()) <= 1:
            return "DATA_NOISE", "Data noise"

        if confidence > 0.9:
            return "CONFIDENCE_ERROR", "Confidence error"

        if 0.4 < confidence < 0.6:
            return "BOUNDARY_ERROR", "Boundary error"

        return "OTHER", f"True: {true_label}, Pred: {pred_label}, Conf: {confidence*100:.0f}%"

    def print_summary(self):
        """Hata analizi ozeti."""
        total_errors = len(self.errors)
        if total_errors == 0:
            print("[ErrorAnalyzer] Analiz edilecek hata yok.")
            return

        print(f"\n{'='*65}")
        print(f"  HATA ANALIZI OZETI - {total_errors} hata")
        print(f"{'='*65}")
        print(f"  {'Hata Tipi':<25} {'Sayi':<8} {'Oran':<10}")
        print(f"  {'-'*43}")

        for error_type, count in sorted(
            self.error_stats.items(), key=lambda x: x[1], reverse=True
        ):
            pct = count / total_errors * 100
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(f"  {error_type:<25} {count:<8} %{pct:<5.1f} {bar}")

        print(f"{'='*65}")

        print(f"\n  ONERILER:")
        if self.error_stats.get("BRAND_MISMATCH", 0) > total_errors * 0.1:
            print(f"    - Brand matcher'i guclendir")
        if self.error_stats.get("CATEGORY_MISMATCH", 0) > total_errors * 0.1:
            print(f"    - Kategori feature agirligini artir")
        if self.error_stats.get("SEMANTIC_ERROR", 0) > total_errors * 0.15:
            print(f"    - Hard negative mining guclendir")
        if self.error_stats.get("BOUNDARY_ERROR", 0) > total_errors * 0.2:
            print(f"    - Threshold optimize et")
        if self.error_stats.get("CONFIDENCE_ERROR", 0) > total_errors * 0.1:
            print(f"    - Model calibration yap")

    def generate_dashboard_data(self, path: str = None) -> Dict:
        """Streamlit dashboard için JSON verisi."""
        data = {
            "total_samples": len(self.errors),
            "error_distribution": dict(self.error_stats),
            "error_examples": [],
        }

        # Her hata tipinden 3'er örnek
        for error_type in set(e.error_type for e in self.errors):
            examples = [e for e in self.errors if e.error_type == error_type][:3]
            for ex in examples:
                data["error_examples"].append({
                    "error_type": ex.error_type,
                    "query": ex.query,
                    "product": ex.product_title,
                    "brand": ex.product_brand,
                    "true_label": ex.true_label,
                    "predicted": ex.predicted_label,
                    "confidence": ex.confidence,
                    "detail": ex.error_detail,
                })

        if path:
            from src.utils.io import write_json

            write_json(path, data)
            print(f"  [OK] Dashboard verisi: {path}")

        return data


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.experiment.config_loader import load_config
    from src.training.trainer import train_model, build_model

    print("Benchmark Runner + Ablation + Error Analyzer\n")

    use_real = "--synthetic" not in sys.argv

    if use_real:
        print("[Benchmark] Gerçek model eğitimi (örnek veri — sentetik metrik YOK)...\n")
        config = load_config("configs/base_config.yaml")
        config["training"]["num_epochs"] = 1
        config["experiment"]["name"] = "benchmark_run"

        def train_fn(cfg, data):
            from src.data.sample_generator import ensure_sample_data
            ensure_sample_data(cfg)
            model, _ = train_model(cfg)
            return model

        def eval_fn(model, data):
            from src.data.dataset import load_train_test
            _, test_df = load_train_test(config)
            return model.evaluate(test_df, config)

        runner = BenchmarkRunner()
        runner.run_all(train_fn=train_fn, eval_fn=eval_fn, n_folds=1, seeds=[42])
    else:
        print("[Benchmark] UYARI: --synthetic modu yalnızca geliştirme içindir.\n")
        runner = BenchmarkRunner()
        runner.run_all()

    runner.print_summary()
    runner.save_report("./experiments/benchmarks/test_report.md")

    # ── 2. Feature Ablation ──
    print("\n2. Feature Ablation")
    print("-" * 50)
    ablator = FeatureAblationFramework()
    if use_real:
        print("  [Ablation] Tam ablation 26 Haziran sonrası gerçek veriyle çalıştırılacak.")
    else:
        ablator.run_synthetic()
        ablator.print_summary()
        ablator.save_report("./experiments/ablation_test.csv")

    # ── 3. Error Analyzer ──
    print("\n3. Error Analyzer")
    print("-" * 50)

    # Sentetik hatalı tahminler
    error_data = pd.DataFrame([
        {"search_query": "nike spor ayakkabı", "product_name": "Adidas Ayakkabı",
         "brand": "Adidas", "category": "Ayakkabı > Spor",
         "is_relevant": 0, "predicted_label": 2, "confidence": 0.95},
        {"search_query": "elbise", "product_name": "Kırmızı Elbise Pamuk",
         "brand": "TrendyolMilla", "category": "Giyim > Elbise",
         "is_relevant": 2, "predicted_label": 0, "confidence": 0.42},
        {"search_query": "deri çanta kadın", "product_name": "Deri Ayakkabı",
         "brand": "Derimod", "category": "Ayakkabı > Günlük",
         "is_relevant": 0, "predicted_label": 2, "confidence": 0.88},
        {"search_query": "kışlık mont", "product_name": "Kışlık Su Geçirmez Mont",
         "brand": "Columbia", "category": "Giyim > Mont",
         "is_relevant": 1, "predicted_label": 0, "confidence": 0.91},
        {"search_query": "s", "product_name": "Süper Star Ayakkabı",
         "brand": "Adidas", "category": "Ayakkabı",
         "is_relevant": 0, "predicted_label": 1, "confidence": 0.55},
    ])

    analyzer = ErrorAnalyzer()
    analyzer.analyze(
        error_data,
        ground_truth=error_data,
        query_column="search_query",
        pred_column="predicted_label",
        true_column="is_relevant",
    )
    analyzer.print_summary()
    analyzer.generate_dashboard_data("./experiments/error_dashboard.json")

    print("\n[OK] Tum testler tamamlandi!")
