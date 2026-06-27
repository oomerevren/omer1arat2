# Validation / Private Leaderboard Simulation Framework

TEKNOFEST Kaggle War Mode için validation omurgası `src_kaggle/validation/` altında kuruldu. Amaç random CV değil; query/term leakage riskini azaltan, OOF-first, threshold-aware, segment raporlu private-LB simülasyonudur.

## Modüller

- `splitters.py`
  - `TermGroupSplitter`
  - `QueryGroupSplitter`
  - `TailAwareSplitter`
- `fold_runner.py`
  - model bağımsız fold akışı
- `oof_manager.py`
  - standart OOF formatı
- `threshold_tuning.py`
  - OOF threshold search ve metrikler
- `segment_reports.py`
  - query/negative/category/gender/age segment skorları
- `seed_stability.py`
  - multi-seed özetleri
- `private_lb_simulator.py`
  - uçtan uca validation run

Script:

```bash
python scripts_kaggle/run_validation.py --config configs/kaggle/war_mode.yaml
```

## Split şemaları

Önerilen ana şema:

```yaml
splitter: term_group
```

Desteklenenler:

- `term_group`
  - aynı `term_id` train ve validation'da birlikte görünmez.
- `query_group`
  - normalized query bazlı leakage kontrolü.
- `tail_aware`
  - tail-query odaklı split için hook; şu an leakage güvenliği için GroupKFold tabanlıdır.

## OOF standardı

OOF çıktısı:

- `id`
- `term_id`
- `item_id`
- `label`
- `query`
- `fold`
- `model_name`
- `proba`
- `pred_default`
- `pred_best_threshold`
- segment flag'leri
- opsiyonel `negative_type`, `source_pool`

OOF dosyası:

```text
reports/validation/oof_predictions.csv
```

## Threshold tuning

`threshold_tuning.py` OOF probability üzerinden 0.05–0.95 aralığında threshold arar.

Raporlananlar:

- macro-F1
- class 0 precision/recall/F1
- class 1 precision/recall/F1
- positive prediction rate
- confusion matrix parçaları

Çıktı:

```text
reports/validation/threshold_curve.csv
```

Threshold seçimi tek fold'a değil birleşik OOF'a dayanır.

## Segment raporları

Desteklenen segmentler:

- `is_short_query`
- `is_long_query`
- `is_brand_heavy`
- `is_attribute_heavy`
- `is_category_heavy`
- `has_gender_token`
- `has_age_token`
- `negative_type`
- `source_pool`
- `category`
- `gender`
- `age_group`

Çıktı:

```text
reports/validation/segment_scores.csv
```

## Seed stability

Config içinde:

```yaml
validation_framework:
  seeds: [42, 2026, 3407]
```

Çıktı:

```text
reports/validation/seed_stability.json
```

Raporlananlar:

- macro-F1 mean/std
- class 0 F1 mean/std
- class 1 F1 mean/std
- threshold mean/std

## Fold-aware negative mining notu

Validation framework config'inde:

```yaml
negative_mode: fold_aware
```

alanı ayrıldı. Mevcut runner, fold split ve model OOF akışını yönetir. Bir sonraki entegrasyon adımı, her fold içinde `build_negatives.py --mode fold_aware --active-fold <fold>` mantığını runner'a bağlamaktır. Bu bağlandığında train fold negatifleri validation fold bilgisine sızmadan üretilecektir.

## Rapor dosyaları

- `reports/validation/validation_summary.json`
- `reports/validation/fold_scores.csv`
- `reports/validation/segment_scores.csv`
- `reports/validation/threshold_curve.csv`
- `reports/validation/seed_stability.json`
- `reports/validation/oof_predictions.csv`

Multi-seed çalışırsa dosyalar `seed_<seed>_` prefix'i ile yazılır.

## Önerilen validation şemaları

1. **V1 term_group**
   - Ana baseline. Aynı term validation'a sızmaz.
2. **V2 query_group**
   - Query normalize edildiğinde aynı query varyantlarının leakage riskini azaltır.
3. **V3 tail_aware**
   - Long-tail davranışı analiz etmek için geliştirilecek hook.
4. **V4 hardness-aware validation**
   - Negative mining metadata'sı ile same-category/same-brand/attribute-conflict alt skorlarını izler.

## Private LB'ye en çok benzeme potansiyeli

Ana öneri:

```text
query_group veya term_group + fold-aware negatives + OOF threshold tuning + segment raporu
```

Eğer aynı query varyantları train/test arasında çok etkiliyse `query_group` daha konservatif ve private-LB'ye daha yakın olabilir. Eğer `term_id` zaten query'yi net temsil ediyorsa `term_group` daha stabil baseline'dır.

## En büyük leakage riskleri

- Negatifleri global üretip sonra split etmek
- Aynı query/term'in train ve validation'da görünmesi
- Retrieval index veya feature'ların validation label bilgisinden etkilenmesi
- Threshold'u tek fold veya public LB üzerinden seçmek
- Ensemble ağırlığını OOF yerine public skorla ayarlamak

## Model kararları nasıl verilmeli?

Bundan sonra model/negative/feature/ensemble kararları şu sırayla verilmelidir:

1. OOF macro-F1
2. class 0 F1
3. segment skorları
4. seed stability
5. threshold stabilitesi
6. ancak en son public LB sinyali

Public LB artıp OOF/segment skorları düşen deneyler riskli kabul edilmelidir.
