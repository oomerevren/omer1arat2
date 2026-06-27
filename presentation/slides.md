# Deep-Pipeline — Teknofest 2026

## Slayt 1: Kapak
- Takım: Deep-Pipeline
- Proje: Türkçe E-Ticaret Ürün-Terim Anlamsal Eşleştirme Sistemi
- Metafor: "Samanlıkta iğne aramıyoruz, iğneyi mıknatısla çekiyoruz."

## Slayt 2: Problem
- Müşteriler farklı kelimelerle aynı ürünü arayabilir.
- E-Ticaret dönüşüm oranı (CVR) arama kalitesine doğrudan bağlıdır.

## Slayt 3: Çözüm Özeti
- BM25 + Vektör Arama hibrit mimari.
- Cross-Encoder tabanlı Reranking.
- Kaggle Macro-F1: 0.8945, Latency: p95 < 250ms.

## Slayt 4: Mimari (Aşama 16)
- Retriever (BM25 + FAISS) -> Reranker (Cross-Encoder) -> Booster (Brand, Category).

## Slayt 5: Hard Negative Mining
- Rastgele negatifler yerine aynı kategorideki alakasız ürünlerle modeli zorlayarak eğittik. (+0.031 F1)

## Slayt 6: Ensemble Stratejisi
- DistilBERTurk + XLM-R + CatBoost.
- Hız ve doğruluğu dengelemek için hafif modeller ve meta-learner kullandık.

## Slayt 7: Performans Metrikleri
- Tam Sistem F1: 0.8945
- TY-embed olmadan F1: 0.8821 (-0.0124)

## Slayt 8: Production ve Hız
- ONNX INT8 Quantization.
- FastAPI + Redis Cache.
- Saniyede >500 sorgu kapasitesi (QPS).

## Slayt 9: Açıklanabilir Yapay Zeka (XAI)
- SHAP Feature etkileri.
- Attention Token haritaları.
- Otomatik doğal dil özetleme.

## Slayt 10: Trendyol Ekosistemine Katkı
- Kullanıma hazır, Türkçe-özel, modüler ve kolay eklenebilir altyapı.

## Slayt 11: Teşekkür
- Dinlediğiniz için teşekkürler! Soru-Cevap.
