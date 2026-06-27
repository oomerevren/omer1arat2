# Project Modes: Kaggle War Mode vs Hackathon / Final Mode

Bu repo artık iki ayrı çalışma modu olarak düşünülmelidir. Ayrımın amacı estetik klasör refactor'ı değil; yarışma sırasında yanlış config, yanlış model modu, yanlış submission ve binary/3-class karışıklığı riskini azaltmaktır.

## 1. Varsayılan mod: Kaggle War Mode

Kaggle için tek ana yol:

```bash
python scripts_kaggle/build_pair_dataset.py --config configs/kaggle/war_mode.yaml
python scripts_kaggle/run_validation.py --config configs/kaggle/war_mode.yaml
python scripts_kaggle/train_baseline.py --config configs/kaggle/war_mode.yaml
python scripts_kaggle/make_submission.py --config configs/kaggle/war_mode.yaml
```

Kaggle problemi sabittir:

- Girdi: `(term_id, item_id)` pair
- Yardımcı tablolar: `terms.csv`, `items.csv`
- Train: sadece pozitif pair'ler
- Test: pozitif + negatif adaylar
- Görev: binary relevance
- Model etiketi: `num_labels=2`
- Metrik: `macro_f1`
- Submission: `id,prediction`

Kaggle script'leri `configs/kaggle/war_mode.yaml` yükler ve config güvenlik kontrolünden geçirir. `mode != kaggle_war_mode`, `num_labels != 2`, `metric != macro_f1` veya hackathon özellikleri açık ise script hata verir.

## 2. Hackathon / Final Mode

Hackathon/final tarafı şu işleri kapsar:

- 3-class logic
- servis/API
- speed optimization
- ONNX / quantization
- explainability UI
- deployment
- demo servisleri

Yeni fiziksel alanlar:

```text
src_hackathon/
  deployment/
  explainability/
  service/
  speed/
  multiclass/
  legacy/

scripts_hackathon/
  run_final_mode_placeholder.py

configs/hackathon/
  final_mode.yaml
```

Eski final/hackathon kodu şimdilik kırılmaması için `src/` ve `scripts/` altında korunmuştur. Ancak Kaggle akışı bu dosyalardan import yapmamalıdır.

## 3. Kaggle mode dosyaları

Kaggle çekirdeği:

```text
src_kaggle/
  data/schema.py
  data/io.py
  data/pairs.py
  data/pair_builder.py
  features/text_features.py
  models/baseline.py
  validation/split.py
  training/pipeline.py
  retrieval/README.md
  inference/submission.py
  utils/config.py
  utils/seed.py

scripts_kaggle/
  build_pair_dataset.py
  run_validation.py
  train_baseline.py
  make_submission.py

configs/kaggle/war_mode.yaml
```

Kaggle submission çıkışı strict olarak `id,prediction` üretir. Eski `product_id/search_query` mantığı bu akışta yoktur.

## 4. Hackathon mode dosyaları

Yeni ayrılmış alan:

```text
src_hackathon/
scripts_hackathon/
configs/hackathon/final_mode.yaml
```

Legacy olarak final/hackathon kabul edilen mevcut dosyalar:

- `src/deployment/`
- `src/xai/`
- `src/inference/onnx_export.py`
- `src/inference/quantizer.py`
- `src/models/cross_encoder.py` ve `num_labels=3` kullanan eski model hattı
- `src/training/`
- `src/retrieval/` servis/search pipeline parçaları
- `scripts/run_dashboard.py`
- `scripts/run_xai_demo.py`
- `scripts/train_cross_encoder.py`
- `scripts/run_experiment.py`
- `scripts/run_ensemble_optuna.py`
- `scripts/run_pseudo_labeling.py`
- `scripts/build_docker.sh`
- `Dockerfile`, `docker-compose.yml`, `monitoring/`

Bu dosyalar silinmedi; Kaggle default yolundan izole edildi.

## 5. Ortak bileşenler

Şimdilik bilinçli olarak minimum ortak kullanım var:

- Python ekosistemi: `pandas`, `numpy`, `scikit-learn`, `pyyaml`
- Veri klasörü: `data/`
- Submission klasörü: `submissions/`
- Dokümantasyon: `docs/`

Kaggle çekirdeği legacy `src/` modüllerini kullanmaz. Ortak yardımcılar ileride gerekiyorsa `src_common/` gibi ayrı bir alana taşınmalıdır; şimdilik yanlış bağımlılık riskini azaltmak için kopyasız ama ayrı tutulmuştur.

## 6. Riskli eski dosyalar

- `configs/base_config.yaml`: Default `mode: final`, `num_labels: 3`, deployment/quantization/pseudo-label gibi Kaggle dışı ayarlar içerir. Kaggle için kullanılmamalı.
- `configs/model/kaggle.yaml`: İsim olarak Kaggle görünse de legacy `base_config.yaml` inherit eder. Deprecated işaretlendi; yeni config `configs/kaggle/war_mode.yaml`.
- `scripts/kaggle_submission.py`: Eski cross-encoder ve `product_id/search_query` submission mantığına bağlı. Yeni Kaggle submission için kullanılmamalı.
- `scripts/run_kaggle_sprint.py`: Eski ensemble/feature pipeline'a bağlı; adı Kaggle olsa bile Prompt 1 çekirdeği değildir.
- `scripts/prepare_kaggle_data.py`: Legacy normalizasyon/config yapısına bağlı. Yeni pair dataset için `scripts_kaggle/build_pair_dataset.py` kullanılmalı.
- `src/models/cross_encoder.py`: Demo/fallback ve yarışma model wrapper riskleri taşıyor; Kaggle'a alınacaksa `src_kaggle/models/` altında binary-only yeniden tasarlanmalı.
- `src/deployment/api.py`: Mock/fallback inference ve servis bağımlılıkları içerir. Kaggle training/submission hattına bağlanmamalı.

## 7. Bilinçli olarak Kaggle çekirdeğinden dışarı alınanlar

- ONNX export
- quantization
- explainability UI
- FastAPI/deployment
- dashboard/demo servisleri
- 3-class final label logic
- product/search-query eski kolon varsayımları
- servis hız optimizasyonları
- demo/fallback model davranışları

## 8. Giriş noktaları

Kaggle:

- `scripts_kaggle/build_pair_dataset.py`
- `scripts_kaggle/run_validation.py`
- `scripts_kaggle/train_baseline.py`
- `scripts_kaggle/make_submission.py`

Hackathon/final:

- `scripts_hackathon/run_final_mode_placeholder.py`
- Legacy: `scripts/` altındaki final/demo/deployment scriptleri; port edilene kadar Kaggle dışıdır.

## 9. Kalan teknik borç

- Legacy `src/` içindeki final kodları henüz fiziksel olarak tamamen taşınmadı; import kırmamak için şimdilik dokümante edilerek izole edildi.
- `src_hackathon/` altında gerçek final implementasyonları placeholder seviyesinde.
- Kaggle için attribute parsing, retrieval, hard-negative mining ve ensembling alanları ayrıldı ama ileri promptlarda güçlendirilmeli.
- Eski `requirements.txt` hâlâ tüm proje bağımlılıklarını içeriyor. Daha sonra `requirements-kaggle.txt` ve `requirements-hackathon.txt` ayrımı yapılmalı.


## 10. Kaggle Data Contract

Gerçek yarışma şeması `docs/kaggle_data_contract.md` içinde sabitlenmiştir. Kaggle kodu legacy `product_*` veya `search_query` kolonlarını kullanmaz.
