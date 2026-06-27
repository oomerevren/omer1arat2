# Kaggle CPU Optimizasyon Planı

Şartname: Kaggle aşamasında GPU sağlanmaz.

## Strateji

1. **DistilBERTurk** birincil model (66M parametre, CPU'da eğitilebilir)
2. **Batch size 32**, **2 epoch** ile hızlı iterasyon
3. **Gradient accumulation** gerektiğinde config'ten artırılır
4. **Negative mining** eğitim öncesi bir kez — epoch başına tekrarlanmaz
5. **FP16 kapalı** CPU'da (`fp16: false` kaggle config override)
6. **Quantization**: Modeller `torch.quantization.quantize_dynamic` (INT8) veya ONNX export ile CPU-optimized formata getirilir.
7. **ONNX Runtime**: `onnxruntime` paketi kullanılarak inference süreleri %30-40 oranında düşürülür.
8. **Batch Tuning**: Kaggle CPU bellek limitine (genellikle 16GB) göre optimal batch_size tespit edilir (Submission batch_size=64 veya 128).

## 26 Haziran Sonrası Checklist

- [ ] `prepare_kaggle_data.py` ile veriyi yerleştir
- [ ] EDA: sütun adları, label dağılımı, null oranları
- [ ] Baseline: `make experiment` (kaggle config)
- [ ] PyTorch Dynamic INT8 ve ONNX modellerinin CPU benchmark testleri
- [ ] Threshold tuning validation split üzerinde
- [ ] Günlük Kaggle submission

## Beklenen Süre (tahmini)

| Adım | CPU süresi |
|------|------------|
| Negative mining | 5–30 dk (veri boyutuna göre) |
| DistilBERTurk 2 epoch | 2–8 saat |
| Model Quantization (INT8/ONNX) | 1-5 dk |
| Submission inference (ONNX) | 5–30 dk |
