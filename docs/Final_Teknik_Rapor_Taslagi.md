# Teknofest 2026 E-Ticaret Hackathonu Final Teknik Rapor Taslağı

## 1. Proje Özeti
Deep-Pipeline, e-ticaret arama süreçlerinde kullanıcı sorgusu ile ürün bilgisini birlikte değerlendiren, açıklanabilir ve yeniden üretilebilir bir ürün–terim ilişkilendirme sistemidir.

## 2. Planlanan / Hazır Teknik Bileşenler
- **Cross-Encoder Pipeline:** DistilBERTurk/BERTurk tabanlı sorgu–ürün çifti sınıflandırma altyapısı hazırlanmıştır. Gerçek yarışma verisi geldiğinde fine-tune edilecektir.
- **Hard Negative Mining:** Benzer fakat alakasız ürünleri seçerek modelin ayrım gücünü artıracak veri üretim modülü hazırlanmıştır.
- **Pseudo-Labeling ve Distillation:** Öğretmen-öğrenci yaklaşımı için kod altyapısı mevcuttur; gerçek performans etkisi veri sonrası ablation ile raporlanacaktır.
- **Açıklanabilir Yapay Zeka (XAI):** Feature katkıları ve attention/overlap tabanlı açıklama katmanı jüri demosu için hazırlanmıştır.
- **Quantization / ONNX Hazırlığı:** Final ortamında CPU performansını iyileştirmek için quantization ve ONNX export fonksiyonları eklenmiştir; nihai latency ölçümü final test ortamında yapılacaktır.

## 3. Performans ve Metrikler
Bu bölüm, yalnızca gerçek organizatör verisi ve final testleri çalıştırıldıktan sonra doldurulacaktır.

- **Macro F1 Score:** [Kaggle public/private leaderboard sonrası doldurulacak]
- **Inference Latency:** [Final donanım/servis testi sonrası doldurulacak]
- **QPS Capacity:** [Stress test sonrası doldurulacak]
- **Ablation Sonuçları:** [Gerçek veriyle yeniden üretilecek]

> Not: Repoda bulunan `*_DEMO.*` dosyaları geçmiş sentetik deneme çıktılarıdır; yarışma performans iddiası olarak kullanılmaz.

## 4. Kullanılan Teknolojiler
Python, pandas, scikit-learn, PyTorch/Transformers, FastAPI, Docker, MLflow, ONNXRuntime, SHAP/LIME ve Türkçe NLP ön işleme bileşenleri.
