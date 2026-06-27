# Deep-Pipeline — Takım Tanıtım Dosyası (KYS Başvuru)

**Yarışma:** TEKNOFEST 2026 E-Ticaret Hackathonu  
**Takım Adı:** Deep-Pipeline  
**Tarih:** 17 Haziran 2026

## Proje Özeti

Deep-Pipeline, e-ticaret arama deneyiminde **ürün–terim ilişkilendirme** problemini çözen uçtan uca bir ML sistemidir. Kullanıcı arama sorgusu ile ürün adı, marka, kategori ve ürün özelliklerini (renk, materyal) birlikte değerlendirerek alaka skoru üretir.

## Teknik Yaklaşım

- **Kaggle aşaması:** DistilBERTurk Cross-Encoder, hard negative mining, Macro-F1 optimizasyonu
- **Final aşaması:** 3 sınıflı sınıflandırma, FastAPI servisi, attention tabanlı XAI, quantization
- **Türkçe NLP:** Zeyrek lemmatization, SymSpell typo düzeltme, RapidFuzz marka eşleştirme

## Mimari Bileşenler

| Bileşen | Açıklama |
|---------|----------|
| Eğitim pipeline | MLflow, CV, distillation, pseudo-labeling |
| API | `/predict`, `/explain`, `/metrics` |
| XAI | Feature + transformer attention açıklamaları |
| Sunum | Next.js 3D arayüz + Streamlit dashboard |

## Özgünlük

Proje sıfırdan bu yarışma için geliştirilmiştir. Negatif örnek üretimi, kategori bazlı eşik optimizasyonu ve Türkçe e-ticaret odaklı ön işleme pipeline'ı takıma özgüdür.

## Beklenen Katkı

Trendyol arama kalitesini artıracak, açıklanabilir ve offline çalışabilir bir relevance modeli sunmayı hedefliyoruz.
