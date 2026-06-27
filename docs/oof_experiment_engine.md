# OOF-first Experiment Engine and Threshold Optimization

Bu modül yarışma kararlarını public LB yerine OOF davranışına dayandırmak için kuruldu.

## Ana scriptler

```bash
python scripts_kaggle/run_experiment.py --config configs/kaggle/war_mode.yaml --name tabular_v1 --model-type tabular
python scripts_kaggle/run_experiment.py --config configs/kaggle/war_mode.yaml --name ce_v1 --model-type cross_encoder
python scripts_kaggle/compare_experiments.py --experiments tabular_v1,ce_v1
```

## Üretilen artefact yapısı

Her deney için:

```text
artifacts/experiments/<experiment_name>/
  oof_predictions.csv
  config_snapshot.json

reports/experiments/<experiment_name>/
  validation/
    oof_predictions.csv
    fold_scores.csv
    segment_scores.csv
    threshold_curve.csv
    seed_stability.json
    validation_summary.json
  threshold/
    threshold_summary.json
    threshold_curve.csv
    fold_thresholds.csv
    segment_thresholds.csv
```

Registry:

```text
reports/experiments/experiment_registry.csv
```

## Standart OOF formatı

Zorunlu kolonlar:

- `id`
- `term_id`
- `item_id`
- `label`
- `fold`
- `proba`
- `pred_default`
- `pred_best_threshold`
- `model_name`
- `experiment_name`

Opsiyonel ama korunabilen kolonlar:

- `negative_type`
- `source_pool`
- query segment flag'leri
- category / gender / age_group

## Experiment registry

`experiment_registry.csv` şu metadata'yı saklar:

- experiment id/name
- model type
- backbone / booster
- data version
- negative mining version
- retrieval version
- feature version
- validation version
- seed
- fold count
- OOF macro-F1
- class 0 F1
- class 1 F1
- best threshold
- threshold fragility
- OOF path
- report dir
- public LB score / not

Bu dosya sıralanabilir ve filtrelenebilir merkezi deney kataloğudur.

## Threshold optimization

Modül:

```text
src_kaggle/validation/threshold_optimizer.py
```

Destekler:

- global threshold sweep
- fold-wise threshold report
- segment-wise threshold analysis
- threshold sensitivity
- class 0 / class 1 F1 trade-off
- predicted positive rate curve

Varsayılan aralık:

```text
0.05 – 0.95, step=0.01
```

Ana karar global OOF threshold üzerinden verilmelidir.

## Segment threshold analizi

Analiz edilen segmentler:

- short query
- brand-heavy query
- attribute-heavy query
- category-heavy query
- same-category / same-brand via `negative_type` or `source_pool`
- attribute-conflict negatives
- gender/age cue segments

Segment threshold çıktısı:

```text
reports/experiments/<experiment>/threshold/segment_thresholds.csv
```

## Segment threshold overfit riski

Segment threshold doğrudan submission stratejisi olarak kullanılmamalı, önce analiz amacıyla görülmelidir.

Risk nerede başlar?

- segment satır sayısı düşükse
- segmentte class distribution dengesizse
- segment threshold global threshold'a göre büyük kazanç gösteriyor ama seed stability düşükse
- threshold çok keskin/narrow optimum gösteriyorsa

Bu yüzden segment raporunda `overfit_risk_note` alanı vardır.

## Model comparison

Modül:

```text
src_kaggle/experiments/model_comparison.py
```

Script:

```bash
python scripts_kaggle/compare_experiments.py --experiments exp_a,exp_b
```

Üretilenler:

- `model_summary.csv`
- `prediction_correlation.csv`
- `model_disagreements.csv`
- model bazlı segment skorları

Amaç yalnızca overall skor değil, ensemble adaylarının farklı hata profillerini görmektir.

## Ensemble için en değerli OOF özellikleri

- düşük prediction correlation
- farklı segmentlerde güçlü performans
- class 0 ve class 1 dengesinde tamamlayıcılık
- disagreement örneklerinde bir modelin net doğru olması
- threshold stabilitesi

## En güvenilir threshold stratejisi

Ana öneri:

```text
Global OOF threshold + threshold sensitivity kontrolü + seed stability
```

Segment threshold şimdilik analiz amaçlıdır. Production/submission için ancak yüksek satır sayısı, düşük seed variance ve tutarlı OOF kazancı varsa düşünülmelidir.

## Public LB nasıl yorumlanmalı?

Public LB karar verici değil, sadece dış sinyal olmalı.

Karar sırası:

1. OOF macro-F1
2. class 0 F1
3. segment skorları
4. threshold sensitivity
5. seed stability
6. model disagreement / ensemble tamamlayıcılığı
7. public LB

Public LB artıyor ama OOF veya class 0 F1 düşüyorsa deney riskli kabul edilmelidir.
