
"""
TEKNOFEST 2026 — Experiment Tracking (MLflow)
===============================================
Otomatik loglama: model, commit, seed, hyperparameter'lar,
macro-F1, threshold, eğitim/inference süresi, model boyutu.

Kullanım:
    tracker = ExperimentTracker("teknofest2026")
    with tracker.start_run("distilbert_baseline", config):
        tracker.log_params(config_dict)
        model = train(...)
        metrics = evaluate(model, ...)
        tracker.log_metrics(metrics)
        tracker.log_model(model, "cross_encoder")
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd

# MLflow opsiyonel import
try:
    import mlflow
    import mlflow.pyfunc
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("[Tracker] MLflow yüklü değil. pip install mlflow")


@dataclass
class ExperimentRun:
    """Tek bir deney çalışmasının tüm metadata'sı."""
    run_id: str
    experiment_name: str
    run_name: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # Model bilgileri
    model_type: str = ""
    model_name: str = ""
    num_params: Optional[int] = None
    model_size_mb: Optional[float] = None
    
    # Git bilgileri
    git_commit: str = ""
    git_branch: str = ""
    
    # Seed
    seed: int = 42
    all_seeds: List[int] = field(default_factory=list)
    
    # Hyperparameter'lar
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Metrikler
    metrics: Dict[str, float] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Threshold
    best_threshold: Optional[float] = None
    
    # Dosya yolları
    artifact_paths: List[str] = field(default_factory=list)


class ExperimentTracker:
    """
    MLflow tabanlı deney takip sistemi.
    
    Ne loglanır:
        - Model adı, tipi, parametre sayısı, boyutu
        - Git commit hash ve branch
        - Seed(ler)
        - Tüm hyperparameter'lar (YAML config'ten)
        - Validation macro-F1, precision, recall
        - Best threshold
        - Eğitim süresi (wall clock)
        - Inference latency (mean, P95)
        - Confusion matrix
        - Model artifact (opsiyonel)
        - Feature importance (opsiyonel)
    
    Kullanım:
        tracker = ExperimentTracker(
            experiment_name="teknofest2026",
            tracking_uri="./experiments/mlruns",
        )
        
        with tracker.start_run("distilbert_exp1", tags={"mode": "kaggle"}) as run:
            tracker.log_config(config_dict)  # tüm YAML config
            tracker.log_hyperparams({"lr": 2e-5, "batch_size": 32})
            
            # Eğitim
            train_result = train_model(...)
            tracker.log_metrics(train_result.metrics)
            tracker.log_duration("training", train_result.duration)
            
            # Inference benchmark
            bench = benchmark_inference(...)
            tracker.log_metrics({"latency_p95": bench.p95, "throughput": bench.qps})
            
            # Model kaydet
            tracker.log_model(model, "cross_encoder_fold0")
    """

    def __init__(
        self,
        experiment_name: str = "teknofest2026",
        tracking_uri: str = None,
        artifact_location: str = None,
        auto_log_system_info: bool = True,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "./experiments/mlruns"
        )
        self.artifact_location = artifact_location
        self.auto_log_system_info = auto_log_system_info
        
        self._active_run: Optional[ExperimentRun] = None
        self._mlflow_available = HAS_MLFLOW
        
        if self._mlflow_available:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._ensure_experiment()
        
        # Git bilgilerini cache'le
        self._git_commit = self._get_git_commit()
        self._git_branch = self._get_git_branch()
    
    def _ensure_experiment(self):
        """Experiment yoksa oluştur."""
        if not self._mlflow_available:
            return
        
        try:
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location,
                )
        except Exception as e:
            print(f"[Tracker] Experiment oluşturma hatası: {e}")
    
    def _get_git_commit(self) -> str:
        """Git commit hash'i al."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _get_git_branch(self) -> str:
        """Git branch adını al."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    @contextmanager
    def start_run(
        self,
        run_name: str = None,
        tags: Dict[str, str] = None,
        nested: bool = False,
    ):
        """
        Context manager olarak deney başlat.
        
        Args:
            run_name: Deney adı. None ise timestamp kullanılır.
            tags: MLflow tags.
            nested: Parent run altında nested run.
        """
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run = ExperimentRun(
            run_id="",
            experiment_name=self.experiment_name,
            run_name=run_name,
            start_time=datetime.now().isoformat(),
            git_commit=self._git_commit,
            git_branch=self._git_branch,
        )
        self._active_run = run
        
        # MLflow run başlat
        if self._mlflow_available:
            try:
                mlflow_run = mlflow.start_run(
                    run_name=run_name,
                    nested=nested,
                    tags=tags,
                )
                run.run_id = mlflow_run.info.run_id
                
                # Otomatik tag'ler
                mlflow.set_tag("git_commit", self._git_commit)
                mlflow.set_tag("git_branch", self._git_branch)
                mlflow.set_tag("start_time", run.start_time)
                
                if tags:
                    for k, v in tags.items():
                        mlflow.set_tag(k, v)
                
            except Exception as e:
                print(f"[Tracker] MLflow başlatma hatası: {e}")
        
        status = "FINISHED"
        try:
            yield run
        except Exception as e:
            status = "FAILED"
            raise e
        finally:
            # End run
            run.end_time = datetime.now().isoformat()
            if run.start_time:
                start_dt = datetime.fromisoformat(run.start_time)
                end_dt = datetime.fromisoformat(run.end_time)
                run.duration_seconds = (end_dt - start_dt).total_seconds()
            
            if self._mlflow_available:
                try:
                    mlflow.set_tag("duration_seconds", str(run.duration_seconds))
                    mlflow.set_tag("end_time", run.end_time)
                    mlflow.end_run(status=status)
                except Exception as e:
                    print(f"[Tracker] MLflow run sonlandırma hatası: {e}")
            
            self._active_run = None
    
    # ================================================================
    # Logging Methods
    # ================================================================
    
    def log_config(self, config: Union[Dict, object], prefix: str = ""):
        """
        YAML config'i flat dict olarak logla.
        Nested dict'leri nokta notasyonuyla düzleştirir.
        
        Örnek:
            {"model": {"name": "bert", "lr": 2e-5}} 
            → {"model.name": "bert", "model.lr": 2e-5}
        """
        if hasattr(config, '__dataclass_fields__'):
            config = asdict(config)
        
        flat = self._flatten_dict(config, prefix)
        
        # MLflow'a logla
        if self._mlflow_available:
            for k, v in flat.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)
        
        # Local run'a da kaydet
        if self._active_run:
            self._active_run.params.update(flat)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """Hyperparameter'ları logla."""
        flat = self._flatten_dict(params, "")
        
        if self._mlflow_available:
            mlflow.log_params(flat)
        
        if self._active_run:
            self._active_run.params.update(flat)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int = None,
        prefix: str = "",
    ):
        """
        Metrikleri logla.
        
        Özel metrikler otomatik etiketlenir:
            - macro_f1 → "validation_macro_f1"
            - latency_p95 → "inference_latency_p95_ms"
        """
        prefixed = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        
        if self._mlflow_available:
            mlflow.log_metrics(prefixed, step=step)
        
        if self._active_run:
            self._active_run.metrics.update(prefixed)
    
    def log_validation_report(
        self,
        report: Any,  # ValidationReport
    ):
        """Gelişmiş validation raporunu logla."""
        if hasattr(report, 'to_dict'):
            report_dict = report.to_dict()
        elif hasattr(report, '__dataclass_fields__'):
            report_dict = asdict(report)
        else:
            report_dict = report
        
        metrics = {}
        if hasattr(report, 'overall_mean_f1'):
            metrics["validation_mean_f1"] = report.overall_mean_f1
            metrics["validation_std_f1"] = report.overall_std_f1
            metrics["validation_stability"] = report.stability_score
        
        self.log_metrics(metrics)
        
        if self._active_run:
            self._active_run.validation_results = report_dict
    
    def log_threshold(self, threshold: float, f1_at_threshold: float):
        """Optimal threshold'u logla."""
        self.log_metrics({
            "best_threshold": threshold,
            "f1_at_best_threshold": f1_at_threshold,
        })
        
        if self._active_run:
            self._active_run.best_threshold = threshold
    
    def log_duration(self, phase: str, seconds: float):
        """Belirli bir fazın süresini logla."""
        self.log_metrics({f"{phase}_duration_seconds": seconds})
    
    def log_model_info(
        self,
        model_type: str,
        model_name: str,
        num_params: int = None,
        model_size_mb: float = None,
    ):
        """Model metadata'sını logla."""
        if self._mlflow_available:
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("model_name", model_name)
            if num_params:
                mlflow.log_param("num_parameters", num_params)
            if model_size_mb:
                mlflow.log_metric("model_size_mb", model_size_mb)
        
        if self._active_run:
            self._active_run.model_type = model_type
            self._active_run.model_name = model_name
            self._active_run.num_params = num_params
            self._active_run.model_size_mb = model_size_mb
    
    def log_model(
        self,
        model: Any,
        artifact_name: str = "model",
        save_fn: callable = None,
    ):
        """
        Model artifact'ini logla.
        
        PyTorch için: torch.save state_dict
        CatBoost için: model.save_model
        Sklearn için: pickle
        """
        if not self._mlflow_available:
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, artifact_name)
            
            if save_fn:
                save_fn(model, path)
            else:
                # Auto-detect
                self._auto_save_model(model, path)
            
            mlflow.log_artifacts(tmpdir, artifact_path=artifact_name)
        
        if self._active_run:
            self._active_run.artifact_paths.append(artifact_name)
    
    def _auto_save_model(self, model, path: str):
        """Model tipine göre otomatik kaydet."""
        import pickle
        
        # PyTorch
        if hasattr(model, 'state_dict'):
            import torch
            torch.save(model.state_dict(), f"{path}.pt")
        # CatBoost
        elif hasattr(model, 'save_model'):
            model.save_model(f"{path}.cbm")
        # LightGBM özel kontrol
        elif type(model).__name__ in ('LGBMClassifier', 'LGBMRegressor', 'Booster'):
            if hasattr(model, 'booster_'):
                model.booster_.save_model(f"{path}.txt")
            elif hasattr(model, 'save_model'):
                model.save_model(f"{path}.txt")
            else:
                with open(f"{path}.pkl", "wb") as f:
                    pickle.dump(model, f)
        # sklearn
        elif hasattr(model, 'get_params'):
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(model, f)
        else:
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(model, f)
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Herhangi bir dosyayı artifact olarak logla."""
        if self._mlflow_available:
            mlflow.log_artifact(local_path, artifact_path)
        
        if self._active_run:
            self._active_run.artifact_paths.append(local_path)
    
    def log_feature_importance(
        self,
        importance: Dict[str, float],
        artifact_name: str = "feature_importance",
    ):
        """Feature importance'ı hem metrics hem artifact olarak logla."""
        # Top 10'u metric olarak
        sorted_fi = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        self.log_metrics({f"fi_{k}": v for k, v in sorted_fi})
        
        # Tamamını JSON artifact olarak
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(importance, f, indent=2)
            self.log_artifact(f.name, artifact_name)
            os.unlink(f.name)
    
    # ================================================================
    # Sorgulama
    # ================================================================
    
    def get_best_run(self, metric: str = "validation_mean_f1") -> Optional[Dict]:
        """En iyi run'ı bul."""
        if not self._mlflow_available:
            return None
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=1,
            )
            
            if len(runs) > 0:
                return runs.iloc[0].to_dict()
        except Exception as e:
            print(f"[Tracker] En iyi run sorgulama hatası: {e}")
        
        return None
    
    def list_runs(self, n: int = 20) -> pd.DataFrame:
        """Son n run'ı DataFrame olarak listele."""
        if not self._mlflow_available:
            return pd.DataFrame()
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return pd.DataFrame()
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=n,
            )
            
            # Robust column fetching
            cols = [
                "run_id", "tags.mlflow.runName",
                "metrics.validation_mean_f1",
                "metrics.validation_std_f1",
                "metrics.training_duration_seconds",
                "metrics.best_threshold",
                "params.model.cross_encoder.model_name",
                "start_time",
            ]
            available = [c for c in cols if c in runs.columns]
            
            # Eğer beklenen metrikler yoksa, olan ilk birkaç metrik ve parametreyi de ekle
            if len(available) <= 3:
                extra_cols = [c for c in runs.columns if c.startswith("metrics.") or c.startswith("params.")]
                available.extend(extra_cols[:5])
                
            return runs[list(set(available))]
        except Exception as e:
            print(f"[Tracker] Run listeleme hatası: {e}")
            return pd.DataFrame()
    
    # ================================================================
    # Helpers
    # ================================================================
    
    def _flatten_dict(
        self, d: Dict, parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Nested dict'i düzleştir."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)):
                # Listeleri string'e çevir
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            elif isinstance(v, (int, float, str, bool, type(None))):
                items.append((new_key, v))
            else:
                items.append((new_key, str(v)))
        
        return dict(items)


# ================================================================
# Hızlı API
# ================================================================

# Singleton tracker
_global_tracker: Optional[ExperimentTracker] = None


def get_tracker(
    experiment_name: str = "teknofest2026",
    tracking_uri: str = None,
) -> ExperimentTracker:
    """Global tracker singleton."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ExperimentTracker(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
        )
    return _global_tracker


# ================================================================
# Test
# ================================================================

if __name__ == "__main__":
    print("Experiment Tracker — Test")
    print("=" * 50)
    
    tracker = ExperimentTracker(
        experiment_name="teknofest2026_test",
        tracking_uri="./experiments/mlruns_test",
    )
    
    with tracker.start_run("test_run", tags={"test": "true"}) as run:
        # Config logla
        config = {
            "model": {"type": "cross_encoder", "name": "distilberturk"},
            "training": {"lr": 2e-5, "epochs": 3},
        }
        tracker.log_config(config)
        
        # Metrik logla
        tracker.log_metrics({
            "validation_macro_f1": 0.0,
            "validation_precision": 0.0,
            "validation_recall": 0.0,
            "training_duration_seconds": 342.5,
        })
        
        # Threshold logla
        tracker.log_threshold(0.65, 0.0)
        
        # Model info
        tracker.log_model_info(
            "cross_encoder",
            "dbmdz/distilbert-base-turkish-cased",
            num_params=66_000_000,
            model_size_mb=265.0,
        )
        
        print(f"  Run ID: {run.run_id}")
        print(f"  Duration: {run.duration_seconds:.1f}s")
        print(f"  Git: {run.git_commit}@{run.git_branch}")
    
    # En iyi run'ı sorgula
    best = tracker.get_best_run("validation_macro_f1")
    if best:
        print(f"\n  Best run F1: {best.get('metrics.validation_macro_f1', 'N/A')}")
    
    print("\n✅ Experiment Tracker testi tamamlandı!")
