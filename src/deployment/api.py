"""
FastAPI deployment — predict + explain + health + metrics.
Offline: LOCAL_MODEL_PATH veya experiments/outputs altından model yükler.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    HAS_PROMETHEUS = True
    REQUEST_COUNT = Counter("search_requests_total", "Total search requests")
    LATENCY_HIST = Histogram("search_latency_seconds", "Search latency")
except ImportError:
    HAS_PROMETHEUS = False


from src.experiment.config_loader import load_config
from src.utils.io import read_json
from src.xai.explainer import ModelExplainer

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

_state: Dict[str, Any] = {
    "model": None,
    "explainer": None,
    "config": None,
    "model_path": None,
    "loaded_at": None,
    "request_count": 0,
    "total_latency_ms": 0.0,
    "validation_f1": 0.0,
    "search_pipeline": None,
}

_cache = {}  # Fallback for Redis

def _get_redis():
    try:
        import redis
        url = os.environ.get("REDIS_URL")
        if url:
            return redis.from_url(url)
    except:
        pass
    return None

redis_client = _get_redis()


class SearchRequest(BaseModel):
    query: str = Field(..., description="Kullanıcı arama sorgusu")
    top_k: int = Field(50, description="Döndürülecek sonuç sayısı")

class PredictRequest(BaseModel):
    search_query: str = Field(..., description="Kullanıcı arama sorgusu")
    product_name: str = Field(..., description="Ürün adı")
    brand: str = ""
    category: str = ""
    product_color: str = ""
    product_material: str = ""


class PredictResponse(BaseModel):
    predicted_label: int
    predicted_class: str
    confidence: float
    probabilities: List[float]
    latency_ms: float


class ExplainResponse(BaseModel):
    query: str
    product_name: str
    prediction: Dict[str, Any]
    features: Dict[str, float]
    feature_explanations: List[Dict[str, Any]]
    summary_tr: str
    visual: Dict[str, Any]
    latency_ms: float


def _find_model_path() -> Optional[Path]:
    candidates = [
        os.environ.get("MODEL_PATH"),
        os.environ.get("LOCAL_MODEL_PATH"),
        "./experiments/outputs/kaggle_baseline/cross_encoder",
        "./experiments/outputs/distilberturk_baseline/cross_encoder",
        "./experiments/outputs/baseline/cross_encoder",
        "./local_model",
    ]
    for c in candidates:
        if c and Path(c).exists() and (Path(c) / "config.json").exists():
            return Path(c)
    return None


def _load_validation_metrics(model_dir: Path) -> float:
    metrics_path = model_dir.parent / "metrics.json"
    if not metrics_path.exists():
        return 0.0
    try:
        data = read_json(metrics_path)
        for key in ("val_macro_f1", "threshold_f1", "macro_f1", "cv_mean_f1"):
            if key in data:
                return float(data[key])
    except (json.JSONDecodeError, OSError, ValueError) as e:
        print(f"[API] UYARI: validation metrikleri okunamadı ({metrics_path}): {e}. F1=0.0 raporlanacak.")
    return 0.0


def _memory_mb() -> float:
    if HAS_PSUTIL:
        return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    return 0.0


def _load_model_and_config():
    config_path = os.environ.get("CONFIG_PATH", "configs/base_config.yaml")
    config = load_config(config_path) if Path(config_path).exists() else {}

    model_path = _find_model_path()
    if model_path:
        from src.models.cross_encoder import CrossEncoderModel
        
        # Check if quantized version exists
        quant_path = model_path.parent / "quantized"
        if config.get("quantization", {}).get("enabled", False) and quant_path.exists():
            print(f"[API] Quantized model bulundu: {quant_path}, yükleniyor (ONNX)...")
            model = CrossEncoderModel.load(model_path) # We load the base for tokenizer
            model.load_onnx(quant_path)
        else:
            model = CrossEncoderModel.load(model_path)
            print(f"[API] Model yüklendi (PyTorch): {model_path}")
            
        _state["validation_f1"] = _load_validation_metrics(model_path)
    else:
        from src.models.cross_encoder import CrossEncoderModel
        from src.data.dataset import get_num_labels

        ce_cfg = config.get("model", {}).get("cross_encoder", {})
        model = CrossEncoderModel(
            model_name=ce_cfg.get("model_name", "dbmdz/distilbert-base-turkish-cased"),
            num_labels=get_num_labels(config) if config else 3,
        )
        print("[API] UYARI: Eğitilmiş model bulunamadı — pretrained ağırlıklarla DEMO modu aktif. Sonuçlar rastgele olabilir.")

    _state["model"] = model
    _state["config"] = config
    _state["model_path"] = str(model_path) if model_path else None
    _state["explainer"] = ModelExplainer(model, config)
    _state["loaded_at"] = time.time()
    
    # Initialize SearchPipeline (mock or real if data available)
    try:
        from src.retrieval.pipeline import SearchPipeline
        from src.retrieval.bm25_index import BM25Index
        from src.retrieval.vector_index import FAISSIndex
        from src.retrieval.reranker import CrossEncoderReranker
        from src.retrieval.boosting import BusinessBooster
        
        # Mock product dataframe
        import pandas as pd
        mock_df = pd.DataFrame({"product_id": ["mock_id_1", "mock_id_2"], "product_name": ["ürün 1", "ürün 2"]})
        
        _state["search_pipeline"] = SearchPipeline(
            bm25_index=BM25Index(), 
            vector_index=FAISSIndex(embedder=None), # Requires real embedder
            reranker=None, booster=None, products_df=mock_df
        )
    except Exception as e:
        print(f"[API] Search pipeline başlatılamadı: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_and_config()
    yield


app = FastAPI(
    title="Deep-Pipeline E-Ticaret Relevance API",
    description="Teknofest 2026 — Ürün-Terim İlişkilendirme ve Açıklanabilirlik Servisi",
    version="1.1.0",
    lifespan=lifespan,
)

if HAS_PROMETHEUS:
    @app.middleware("http")
    async def track_metrics(request: Request, call_next):
        if request.url.path in ["/search", "/predict", "/explain"]:
            REQUEST_COUNT.inc()
            import time
            start = time.time()
            response = await call_next(request)
            LATENCY_HIST.observe(time.time() - start)
            return response
        return await call_next(request)

    @app.get("/prometheus_metrics")
    def get_prometheus_metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_path": _state.get("model_path"),
        "mode": _state["config"].get("experiment", {}).get("mode", "unknown") if _state["config"] else "demo",
    }


@app.get("/metrics")
def metrics():
    count = _state["request_count"]
    avg_lat = _state["total_latency_ms"] / max(count, 1)
    model_lat = 0.0
    if _state["model"] and hasattr(_state["model"], "_last_inference_ms"):
        model_lat = _state["model"]._last_inference_ms
    effective_lat = avg_lat if count > 0 else model_lat
    return {
        "total_requests": count,
        "avg_latency_ms": round(avg_lat, 2),
        "inference_latency_ms": round(effective_lat, 2),
        "memory_mb": _memory_mb(),
        "macro_f1": round(_state.get("validation_f1", 0.0), 4),
        "qps_capacity": max(1, int(1000 / max(effective_lat, 1))),
        "demo_mode": _state.get("model_path") is None,
    }


@app.post("/search")
def search(req: SearchRequest):
    t0 = time.perf_counter()
    cache_key = f"search:{req.query.lower()}:{req.top_k}"
    
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    else:
        if cache_key in _cache:
            return _cache[cache_key]

    # Mock search response since pipeline is not fully initialized with real data here
    # In a real scenario, we would use _state["search_pipeline"].search(req.query, req.top_k)
    results = [("mock_id_1", 0.95), ("mock_id_2", 0.88)]
    
    latency = (time.perf_counter() - t0) * 1000
    
    response = {
        "query": req.query,
        "results": [
            {"product_id": pid, "score": float(score)} for pid, score in results
        ],
        "latency_ms": round(latency, 2)
    }
    
    if redis_client:
        redis_client.setex(cache_key, 900, json.dumps(response))
    else:
        _cache[cache_key] = response
        
    return response

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(503, "Model henüz yüklenmedi.")

    t0 = time.perf_counter()
    result = _state["model"].predict_single(
        req.search_query,
        req.product_name,
        _state["config"],
        brand=req.brand,
        category=req.category,
        color=req.product_color,
        material=req.product_material,
    )
    latency = (time.perf_counter() - t0) * 1000
    _state["request_count"] += 1
    _state["total_latency_ms"] += latency

    return PredictResponse(
        predicted_label=result["predicted_label"],
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        latency_ms=round(latency, 2),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest):
    if _state["explainer"] is None:
        raise HTTPException(503, "Explainer henüz yüklenmedi.")

    t0 = time.perf_counter()
    result = _state["explainer"].explain(
        req.search_query,
        req.product_name,
        brand=req.brand,
        category=req.category,
        color=req.product_color,
        material=req.product_material,
    )
    latency = (time.perf_counter() - t0) * 1000
    _state["request_count"] += 1
    _state["total_latency_ms"] += latency

    return ExplainResponse(
        query=result["query"],
        product_name=result["product_name"],
        prediction=result["prediction"],
        features=result["features"],
        feature_explanations=result["feature_explanations"],
        summary_tr=result["summary_tr"],
        visual=result["visual"],
        latency_ms=round(latency, 2),
    )
