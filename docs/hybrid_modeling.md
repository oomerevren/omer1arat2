# Hybrid Modeling System

TEKNOFEST Kaggle War Mode için OOF-first hibrit modelleme çatısı kuruldu. Sistem üç aileyi destekler:

1. Tabular / feature-based model
2. Cross-encoder / semantic pair model
3. OOF tabanlı weighted ensemble

## Modüller

- `src_kaggle/models/tabular_model.py`
- `src_kaggle/models/cross_encoder_model.py`
- `src_kaggle/models/pair_text_builder.py`
- `src_kaggle/models/ensemble.py`
- `src_kaggle/models/oof.py`
- `src_kaggle/training/train_tabular.py`
- `src_kaggle/training/train_cross_encoder.py`
- `src_kaggle/training/train_ensemble.py`

Scriptler:

- `scripts_kaggle/train_tabular.py`
- `scripts_kaggle/train_cross_encoder.py`
- `scripts_kaggle/train_ensemble.py`

## Tabular model

Tabular model feature engineering çıktısını kullanır. Öncelik hızlı ve güvenli OOF deneyidir.

Desteklenen backendler:

- `hist_gradient_boosting` default, sklearn CPU-safe
- `lightgbm` varsa kullanılır
- `catboost` varsa kullanılır

Çıktılar:

- `artifacts/models/tabular/tabular_fold{fold}.joblib`
- `artifacts/oof/tabular_oof.csv`
- `reports/models/tabular_cv_report.json`
- `reports/models/tabular_feature_importance.csv`

## Cross-encoder / semantic pair model

Dosya:

- `src_kaggle/models/cross_encoder_model.py`

Text format builder ayrı tutuldu:

- `src_kaggle/models/pair_text_builder.py`

Desteklenen formatlar:

- `query_title`
- `query_title_category`
- `query_title_category_brand`
- `full_v1`

`full_v1` örneği:

```text
[QUERY] nike beyaz kadın sneaker [SEP] [TITLE] nike air beyaz sneaker [SEP] [CATEGORY] ayakkabı/sneaker [SEP] [BRAND] nike [SEP] [ATTR] color beyaz material tekstil [SEP] [GENDER] kadın [SEP] [AGE] yetişkin
```

Backendler:

- `sklearn_text`: CPU-safe TF-IDF + LogisticRegression pair-text classifier
- `transformers`: gerçek GPU fine-tuning için arayüz ayrıldı; demo/fallback olarak sessiz çalışmaz, bilinçli implementasyon gerektirir

Çıktılar:

- `artifacts/models/cross_encoder/cross_encoder_fold{fold}.joblib`
- `artifacts/oof/cross_encoder_oof.csv`
- `reports/models/cross_encoder_cv_report.json`

## Ensemble

Dosya:

- `src_kaggle/models/ensemble.py`

Desteklenen yöntem:

- OOF tabanlı weighted average

Çıktılar:

- `artifacts/models/ensemble/weighted_average.joblib`
- `reports/models/ensemble_cv_report.json`

Ensemble threshold OOF üzerinden seçilir. Public LB skoruna göre ağırlık seçme yaklaşımı özellikle önerilmez.

## OOF standardı

OOF dosyaları ortak kolon yapısına yakındır:

- `id`
- `term_id`
- `item_id`
- `label`
- `fold`
- `model_name`
- `proba`

OOF şu işler için zorunludur:

- threshold tuning
- blending
- model karşılaştırma
- overfit kontrolü
- private LB simülasyonu

## Config

`configs/kaggle/war_mode.yaml` içinde `modeling` bloğu eklendi.

Özet:

```yaml
modeling:
  oof_first: true
  tabular:
    model_type: hist_gradient_boosting
    n_folds: 5
  cross_encoder:
    backend: sklearn_text
    model_name: dbmdz/distilbert-base-turkish-cased
    text_format_version: full_v1
  ensemble:
    method: weighted_average
    blend_weights:
      tabular: 0.5
      cross_encoder: 0.5
    threshold_source: oof
```

## Hangi model hangi sinyalde güçlü?

### Tabular model

Güçlü olduğu sinyaller:

- lexical overlap
- category overlap
- brand mismatch / contradiction
- attribute exact match / conflict
- gender / age conflict
- retrieval scores
- semantic numeric scores

Özellikle class 0 F1 için güçlüdür çünkü net çelişki sinyallerini sayısal olarak öğrenebilir.

### Cross-encoder / pair text model

Güçlü olduğu sinyaller:

- query-title semantik yakınlığı
- query ile item açıklamasının birlikte yorumlanması
- kelime sırası / bağlam
- exact olmayan ama anlam olarak yakın eşleşmeler

Transformer backend eklendiğinde semantik güç artar. Şu an CPU-safe `sklearn_text` backend OOF altyapısını test etmek ve text-format ablation yapmak için kullanılabilir.

### Ensemble

Tek modelden daha güvenlidir çünkü:

- tabular model exact/attribute çelişkilerinde güçlüdür
- semantic pair model text relevance tarafında güçlüdür
- hata profilleri farklıdır
- OOF blend, public LB yerine validation davranışına göre karar verir

## Overfit riski taşıyan blend stratejileri

- Public LB skoruna göre manuel ağırlık seçmek
- OOF olmadan test prediction blend etmek
- Çok fazla modelle stacking yapmak
- Negative mining artifact'lerini aşırı öğrenen tabular modelle yüksek ağırlık vermek
- Fold-aware olmayan retrieval feature'larıyla ensemble yapmak

## Sonraki entegrasyon

1. Validation framework group/fold split ile netleştirilmeli.
2. Negative mining fold-aware modda her fold için çalıştırılmalı.
3. Threshold tuning modülü OOF dosyalarını okuyarak segment bazlı eşik seçmeli.
4. Cross-encoder için gerçek transformer Trainer backend GPU ortamında eklenmeli.
5. Ensemble ağırlıkları OOF macro-F1 ve segment davranışı üzerinden optimize edilmeli.
