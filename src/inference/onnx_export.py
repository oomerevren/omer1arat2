import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def export_cross_encoder_to_onnx(model_path: str, onnx_path: str, max_length: int = 256):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    dummy = tokenizer("test query", "test product", 
                       return_tensors="pt", max_length=max_length,
                       padding="max_length", truncation=True)
                       
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                       
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = onnx_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)
        print(f"ONNX Quantization completed: {quantized_path}")
        return quantized_path
    except ImportError:
        print("onnxruntime not installed. Skipping quantization.")
        return onnx_path

if __name__ == "__main__":
    # Example usage
    pass
