"""
Model quantization — dynamic INT8 ve ONNX export.

Torch opsiyoneldir. Torch yoksa fonksiyonlar güvenli şekilde no-op davranır; böylece
CI/import smoke testleri ağır bağımlılıklar olmadan da geçer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def quantize_dynamic(model, save_path: Optional[str] = None):
    """PyTorch dynamic quantization (INT8). Torch yoksa modeli değiştirmeden döndürür."""
    if not HAS_TORCH or not hasattr(model, "state_dict"):
        print("[Quantizer] Torch/model yok; dynamic quantization atlandı.")
        return model
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized.state_dict(), save_path)
    return quantized


def export_onnx(
    model,
    tokenizer,
    save_path: str,
    max_length: int = 256,
    opset: int = 14,
) -> str:
    """Cross-encoder'ı ONNX formatına export eder."""
    if not HAS_TORCH:
        raise RuntimeError("ONNX export için torch gerekli.")
    try:
        import torch.onnx
    except ImportError:
        raise RuntimeError("ONNX export için torch.onnx gerekli.")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = tokenizer(
        "örnek sorgu",
        "örnek ürün",
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        save_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )
    return save_path


def apply_quantization(config: Dict[str, Any], model, output_dir: str = "./experiments/quantized") -> Dict[str, str]:
    """Config'e göre quantization uygular. Torch yoksa boş dict döner."""
    q_cfg = config.get("quantization", {})
    if not q_cfg.get("enabled", False):
        return {}
    if not HAS_TORCH:
        print("[Quantizer] Torch yüklü değil; quantization atlandı.")
        return {}

    method = q_cfg.get("method", "dynamic")
    paths = {}

    if method == "dynamic":
        path = os.path.join(output_dir, "model_int8.pt")
        quantize_dynamic(model, path)
        paths["int8"] = path
    elif method == "onnx":
        path = os.path.join(output_dir, "model.onnx")
        # ONNX export: tokenizer ve model erişilebilir olmalı (CrossEncoderModel wrapper)
        if hasattr(model, "tokenizer") and hasattr(model, "model"):
            export_onnx(
                model.model,
                model.tokenizer,
                path,
                max_length=getattr(model, "max_length", 256),
                opset=q_cfg.get("onnx_opset", 14),
            )
        elif hasattr(model, "save_pretrained"):
            # Raw HF model: tokenizer olmadan export edilemez — trainer'dan wrapper ile çağrılmalı
            print("[Quantizer] UYARI: ONNX export için tokenizer gerekli. Lütfen CrossEncoderModel wrapper'ı ile çağırın.")
        paths["onnx"] = path

    return paths
