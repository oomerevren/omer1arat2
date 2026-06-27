# TEKNOFEST 2026 Kaggle War Mode

Bu doküman repodaki Kaggle aşamasının tek ve net ana yolunu tanımlar. Amaç, genel ürün/hackathon sistemi değil; private leaderboard odaklı, hızlı deney yapılabilen binary pair-classification hattıdır.

## 1. Scope

### Kaggle War Mode problemi

- Girdi satırı: `(term_id, item_id)` çifti
- Yardımcı tablolar: `terms.csv`, `items.csv`
- Eğitim verisi: yalnızca pozitif çiftler
- Test verisi: pozitif + negatif aday çiftler
- Tahmin: binary relevance `0/1`
- Metrik: `macro-F1`

### Şu anda geliştirilen mod

Yalnızca **Kaggle War Mode**. Hackathon/final modu korunur ama Kaggle ana akışına dahil değildir.

## 2. Tek ana Kaggle pipeline

Kaggle için canonical yol aşağıdaki komut sırasıdır:

```bash
python scripts_kaggle/build_pair_dataset.py \
  --config configs/kaggle/war_mode.yaml \
  --train data/training_pairs.csv \
  --test data/submission_pairs.csv \
  --terms data/terms.csv \
  --items data/items.csv \
  --out-train data/kaggle/train_pairs_features.csv \
  --out-test data/kaggle/submission_pairs_features.csv

python scripts_kaggle/run_validation.py \
  --config configs/kaggle/war_mode.yaml \
  --train-pairs data/kaggle/train_pairs_features.csv

python scripts_kaggle/train_baseline.py \
  --config configs/kaggle/war_mode.yaml \
  --train-pairs data/kaggle/train_pairs_features.csv \
  --model-out models_kaggle/tfidf_logreg.joblib \
  --metrics-out models_kaggle/metrics.json

python scripts_kaggle/make_submission.py \
  --config configs/kaggle/war_mode.yaml \
  --test-pairs data/kaggle/submission_pairs_features.csv \
  --model models_kaggle/tfidf_logreg.joblib \
  --metrics models_kaggle/metrics.json \
  --output submissions/kaggle_submission.csv
```

Bu dört script Kaggle çekirdeğinin dışarıdan görünen tek giriş noktasıdır.

## 3. Güncel klasör yapısı

```text
src_kaggle/
  data/
    io.py              # terms/items/pair CSV okuma-yazma ve text join
    pairs.py           # pozitif train çiftlerinden binary pair dataset üretimi
    schema.py          # canonical kolon isimleri
  features/
    text_features.py   # TF-IDF pair text özellikleri
  models/
    baseline.py        # TF-IDF + LogisticRegression baseline
  validation/
    split.py           # stratified fold üretimi
  training/
    pipeline.py        # cross-validation, threshold search, full train
  retrieval/
    README.md          # ileride hard-negative/candidate analizi için ayrılmış alan
  inference/
    submission.py      # submission dataframe üretimi
  utils/
    seed.py            # determinism helper

scripts_kaggle/
  build_pair_dataset.py
  train_baseline.py
  run_validation.py
  make_submission.py

docs/
  kaggle_war_mode.md
```

## 4. Kaggle core listesi

Bu dosyalar Kaggle War Mode çekirdeğidir ve skor deneyleri burada ilerlemelidir:

- `src_kaggle/data/schema.py`
- `src_kaggle/data/io.py`
- `src_kaggle/data/pairs.py`
- `src_kaggle/features/text_features.py`
- `src_kaggle/models/baseline.py`
- `src_kaggle/validation/split.py`
- `src_kaggle/training/pipeline.py`
- `src_kaggle/inference/submission.py`
- `src_kaggle/utils/seed.py`
- `scripts_kaggle/build_pair_dataset.py`
- `scripts_kaggle/run_validation.py`
- `scripts_kaggle/train_baseline.py`
- `scripts_kaggle/make_submission.py`

## 5. Hackathon later / ikinci planda kalacaklar

Aşağıdaki bileşenler silinmedi; ancak Kaggle ana akışında kullanılmamalıdır:

- `src/deployment/` ve FastAPI servis mantığı
- `src/inference/onnx_export.py`
- `src/inference/quantizer.py`
- `src/xai/`
- `scripts/run_dashboard.py`
- `scripts/run_xai_demo.py`
- `scripts/build_docker.sh`
- `Dockerfile`, `docker-compose.yml`, `monitoring/`
- rapor/presentasyon üretim dosyaları
- 3-class hackathon label/servis mantığı içeren eski eğitim-inference hattı

Bu parçalar final/hackathon gösterimi, servisleşme, açıklanabilirlik veya deployment için daha sonra değerlendirilebilir.

## 6. Eski dosyalara dair risk notları

- `scripts/kaggle_submission.py`: Eski `src` cross-encoder hattına bağlı. Kolon varsayımları `product_id/search_query` odaklı; yeni Kaggle problemi `term_id/item_id` pair classification olduğu için ana Kaggle yolu olmamalı.
- `scripts/prepare_kaggle_data.py`: Eski config/dataset normalizasyonuna bağlı. Yeni Kaggle schema netliği için `scripts_kaggle/build_pair_dataset.py` tercih edilmeli.
- `scripts/run_kaggle_sprint.py`: İsim olarak Kaggle odaklı görünse de eski ensemble ve feature pipeline'a bağlı; Prompt 0 çekirdeğine dahil değil.
- `src/models/cross_encoder.py`, `src/models/ensemble.py`, `src/training/trainer.py`: Değerli deney alanları olabilir; ancak mevcut halleri genel/hackathon mimarisiyle karışık. Kaggle'a taşınacaksa `src_kaggle/` altında sadeleştirilerek taşınmalı.
- `src/retrieval/`: Arama/servis mantığı içeriyor. Kaggle test dosyası zaten `(term_id,item_id)` adayları verdiği için ana submission hattına retrieval zorunlu değildir.
- `src/inference/onnx_export.py`, `src/inference/quantizer.py`: Yarışma skorunu doğrudan artırmayan servis optimizasyonlarıdır; şimdilik dışarıda kalmalı.
- `src/xai/explainer.py`: XAI ve demo faydası olabilir ama private leaderboard odaklı iterasyonu yavaşlatabilir.

## 7. Başarı hedefi

Madde 0'ın başarı kriteri skor artışı değil; sonraki tüm skor odaklı düzeltmelerin güvenli ve hızlı yapılacağı savaş alanını kurmaktır. Geliştirici repoya baktığında Kaggle için çalıştıracağı yol:

1. `scripts_kaggle/build_pair_dataset.py`
2. `scripts_kaggle/run_validation.py`
3. `scripts_kaggle/train_baseline.py`
4. `scripts_kaggle/make_submission.py`

şeklinde 1 dakika içinde anlaşılmalıdır.


Veri sözleşmesi: [`docs/kaggle_data_contract.md`](kaggle_data_contract.md)
