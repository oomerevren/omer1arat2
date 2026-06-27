# Error Analysis + Controlled Pseudo Labeling + Experiment Tracking

Bu aşamada hata analizi, pseudo-labeling ve tracking tek karar sistemine bağlandı.

## Error analysis

Modüller:

- `src_kaggle/analysis/error_taxonomy.py`
- `src_kaggle/analysis/error_analysis.py`
- `scripts_kaggle/run_error_analysis.py`

Komut:

```bash
python scripts_kaggle/run_error_analysis.py --oof artifacts/experiments/<exp>/oof_predictions.csv --threshold 0.5
```

Çıktılar:

- `reports/errors/top_false_positives.csv`
- `reports/errors/top_false_negatives.csv`
- `reports/errors/annotated_errors.csv`
- `reports/errors/error_summary.json`
- `reports/errors/error_summary.md`

Desteklenen hata etiketleri:

- `brand_mismatch`
- `category_mismatch`
- `attribute_mismatch`
- `gender_conflict`
- `age_group_conflict`
- `title_ambiguity`
- `query_ambiguity`
- `synonym_vocabulary_issue`
- `typo_spelling_issue`
- `missing_attribute_issue`
- `false_negative_sampling_issue`
- `retrieval_source_issue`
- `threshold_error`
- `calibration_error`

Rapor aksiyon önerisi de üretir: attribute feature güçlendirme, same-brand negative artırma, threshold kalibrasyonu, retrieval pool iyileştirme vb.

## Controlled pseudo-labeling

Modüller:

- `src_kaggle/pseudo_labeling/confidence_filters.py`
- `src_kaggle/pseudo_labeling/pseudo_labeler.py`
- `scripts_kaggle/run_pseudo_labeling.py`

Komut:

```bash
python scripts_kaggle/run_pseudo_labeling.py \
  --config configs/kaggle/war_mode.yaml \
  --candidates data/processed/pseudo_candidates.parquet
```

Modlar:

- `disabled`
- `positive_only`
- `negative_only`
- `dual`

Güvenlik kuralları:

Pozitif pseudo-label için:

- probability çok yüksek olmalı
- margin yüksek olmalı
- model agreement yeterli olmalı
- lexical / retrieval / semantic support sinyali olmalı

Negatif pseudo-label için:

- probability çok düşük olmalı
- margin yüksek olmalı
- conflict veya düşük lexical overlap gibi güvenli negatif sinyali olmalı

Çıktılar:

- `data/processed/pseudo_labels.parquet`
- `reports/pseudo_labeling/pseudo_label_report.json`
- `reports/pseudo_labeling/pseudo_label_report.md`

Risk raporunda:

- pseudo count
- positive/negative count
- segment distribution
- agreement summary
- margin summary
- risk notes

bulunur.

## Experiment tracking

Modül:

- `src_kaggle/tracking/experiment_tracker.py`

Çıktılar:

- `reports/experiments/master_experiment_log.csv`
- `reports/experiments/master_experiment_log.md`

Loglanan kritik metadata:

- run id
- timestamp
- git commit
- experiment name
- model type
- data / negative mining / retrieval / feature / validation version
- seed
- OOF macro-F1
- class 0 F1
- class 1 F1
- threshold
- pseudo-labeling mode
- pseudo-label count
- config path
- OOF path
- error report path
- pseudo report path
- artifact dir

## Pseudo-label ne zaman güvenli?

Pseudo-labeling ancak şu koşullarda güvenli kabul edilmeli:

1. OOF macro-F1 ve class 0 F1 stabilse
2. threshold sensitivity düşükse
3. seed stability iyiyse
4. ensemble/model agreement yüksekse
5. retrieval/lexical/semantic support sinyalleri aynı yöndeyse
6. pseudo-label hacmi kontrollü ve segment dağılımı makulse

## Tracking'te olmazsa olmaz metadata

- experiment id/name
- git/code version
- data version
- negative mining version
- feature version
- validation setup
- seed/fold count
- OOF macro-F1
- class 0/class 1 F1
- threshold
- OOF path
- pseudo-label mode/count
- error analysis report

## Deney seçim sinyalleri

Bundan sonra deney seçimi şu sinyallere göre yapılmalı:

1. OOF macro-F1
2. class 0 F1
3. segment error reduction
4. threshold stability
5. seed stability
6. error tag dağılımında iyileşme
7. model disagreement/ensemble faydası
8. public LB sadece son dış sinyal

## Açık not

Gerçek veri üzerinde en sık hata tipleri ancak OOF çıktıları üretildikten sonra raporlanabilir. Sistem şu an bu etiketleri otomatik üretmeye hazırdır.
