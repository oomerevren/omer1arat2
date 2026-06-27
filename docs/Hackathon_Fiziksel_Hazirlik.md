# Teknofest 2026 Hackathon Fiziksel Çadır Hazırlığı

Bu doküman, Teknofest E-Ticaret Hackathonu final çadırında (offline ortamda) takımın yaşayabileceği sorunlara karşı alınan fiziksel ve yazılımsal tedbirleri listeler.

## 1. Donanım ve İnternetsiz Ortam (Offline Desteği)
Yarışma çadırında internet bağlantısı sınırlandırılabilir veya tamamen kesilebilir. Bu duruma karşı hazırlıklar:
- **Lokal Model Klasörü**: `local_model/` dizinine kullanılacak tüm transformer modelleri (BGE-M3, DistilBERTurk) yarışma alanına gitmeden önce indirilmiş olmalıdır.
- **Docker İmajı**: `deep-pipeline` Docker imajı, tüm Python kütüphaneleri (PyTorch, Transformers, ONNX vb.) dahil olacak şekilde önceden derlenmiş ve test edilmiş olmalıdır (`docker save` ve `docker load` ile taşınabilir formatta).
- **Zemberek ve Zeyrek**: Dil işleme modelleri ve sözlük dosyaları internetsiz çalışacak şekilde önbelleğe alınmıştır.
- **Bağımlılıklar (Pip)**: `requirements.txt` içindeki paketler teker teker offline indirilip wheel dosyaları yedeklenebilir.

## 2. Çadır İçi Görev Dağılımı
- **Kişi 1 (Model ve Optimizasyon)**: Modelin (Cross-Encoder) offline eğitimi, veri temizleme ve submission çıktılarını üretmekten sorumludur.
- **Kişi 2 (Sunum ve XAI)**: Streamlit Dashboard'unun canlı tutulması, elde edilen sonuçların jüri için görselleştirilmesi ve sunum/web arayüzü optimizasyonunu yönetir.
- **Kişi 3 (Pipeline & Hata Giderme)**: Docker, ONNX entegrasyonu, sistem hatalarının (RAM limitleri, crash) giderilmesi ve submission sürelerinin optimizasyonu (Batch Tuning).

## 3. Donanım Planlaması
- **Makineler**: Minimum 16GB, tercihen 32GB RAM'li dizüstü bilgisayarlar (CPU tabanlı Kaggle submission için).
- **Disk**: En az 100 GB boş alanlı SSD disk.
- **Offline Yedekleme**: Tüm kodlar, docker imajı ve model dosyaları taşınabilir harici disklere veya USB belleklere çift yedeklenmiş şekilde çadıra getirilmelidir.

## 4. Acil Durum Çantası (Yazılımsal)
- RAM tükenme (OOM) ihtimaline karşı `batch_size` 16 veya 8'e çekilmeli.
- Zaman daraldığında 2 epoch yerine 1 epoch DistilBERTurk eğitip geçiş yapılmalı.
- ONNX çalışmazsa fallback olarak standart PyTorch CPU inference moduna geri dönülmeli.
