# DEEP-PiPELiNE — TEKNOFEST 2026 E-Ticaret Kaggle War Mode

Bu repo TEKNOFEST 2026 E-Ticaret Kaggle aşaması için pair-centric, OOF-first, private leaderboard odaklı yarışma pipeline'ıdır.

## Problem

Görev `(term_id, item_id)` çifti için binary relevance tahminidir.

- Train: `training_pairs.csv` yalnızca pozitif `label=1`
- Test: `submission_pairs.csv` pozitif + negatif aday çiftler
- Yardımcı tablolar: `terms.csv`, `items.csv`
- Submission: sadece `id,prediction`
- Metrik: macro-F1

## Official data contract

Dosyalar `data/` altına konur:

```text
data/items.csv              # item_id,title,category,brand,gender,age_group,attributes
data/terms.csv              # term_id,query
data/training_pairs.csv     # id,term_id,item_id,label
data/submission_pairs.csv   # id,term_id,item_id
data/sample_submission.csv  # id,prediction
```

## Quickstart / Reproduction

```bash
# 1) Pair-centric merged dataset
python scripts_kaggle/build_pair_dataset.py --config configs/kaggle/war_mode.yaml

# 2) Retrieval index ve rapor
python scripts_kaggle/build_retrieval_index.py --config configs/kaggle/war_mode.yaml

# 3) Hard negatives + augmented train
python scripts_kaggle/build_negatives.py --config configs/kaggle/war_mode.yaml

# 4) Feature matrix
python scripts_kaggle/build_features.py --config configs/kaggle/war_mode.yaml

# 5) OOF-first validation / experiment
python scripts_kaggle/run_experiment.py --config configs/kaggle/war_mode.yaml --name tabular_v1 --model-type tabular

# 6) Error analysis
python scripts_kaggle/run_error_analysis.py --oof artifacts/experiments/tabular_v1/oof_predictions.csv

# 7) Safe submission
python scripts_kaggle/make_submission.py --config configs/kaggle/war_mode.yaml --experiment-id tabular_v1 --models-used tabular
```

## Ana pipeline bileşenleri

### Pair-centric data pipeline

`training_pairs/submission_pairs + terms + items` güvenli join edilir. Çıktılar:

```text
data/processed/train_pairs_merged.parquet
data/processed/submission_pairs_merged.parquet
reports/data_quality/*pair_build_report.json
```

### Negative mining

Çok katmanlı negatif üretimi:

- easy
- same-category
- same-brand
- lexical confusing
- attribute conflict
- dense-hard için hazır hook

Çıktı:

```text
data/processed/train_with_negatives.parquet
reports/negative_mining/negative_mining_report.json
```

### Retrieval

BM25 + dense fallback + hybrid candidate pools:

```text
artifacts/retrieval/hybrid_retriever.pkl
reports/retrieval/index_report.json
```

### Feature engineering

Lexical, category, brand, attribute, gender/age, query intent, retrieval, semantic ve metadata feature'ları üretir.

```text
data/processed/train_features.parquet
reports/features/feature_catalog.json
```

### Modeling

OOF-first hibrit modelleme:

- tabular booster / sklearn HGB fallback
- cross-encoder pair text backend
- weighted OOF ensemble

### Validation

Private-LB simülasyonu:

- term/query group split
- OOF threshold tuning
- segment raporları
- seed stability

### Error analysis / pseudo-labeling / tracking

- error taxonomy ve aksiyon önerileri
- kontrollü pseudo-labeling
- experiment registry ve master log

### Submission safety

Submission yalnızca `id,prediction` üretir ve strict validator'dan geçmeden final dosya yazılmaz.

## Repo yapısı

```text
src_kaggle/              # Kaggle War Mode çekirdeği
scripts_kaggle/          # tek resmi Kaggle entrypoint scriptleri
configs/kaggle/          # Kaggle config
docs/                    # teknik tasarım belgeleri
reports/                 # rapor çıktıları
artifacts/               # model/index/experiment artefactları
src_hackathon/           # final/hackathon için ayrılmış alan
scripts_hackathon/       # final/hackathon script alanı
src/                     # legacy/final eski kodlar; Kaggle ana yolu değildir
```

## Ana teknik dokümanlar

- [Pipeline Overview](docs/kaggle_pipeline_overview.md)
- [Data Contract](docs/data_contract.md)
- [Negative Mining Design](docs/negative_mining_design.md)
- [Retrieval Design](docs/retrieval_design.md)
- [Feature Engineering Design](docs/feature_engineering_design.md)
- [Modeling Design](docs/modeling_design.md)
- [Validation Strategy](docs/validation_strategy.md)
- [Submission Process](docs/submission_process.md)
- [OOF Experiment Engine](docs/oof_experiment_engine.md)
- [Error/Pseudo/Tracking](docs/error_pseudo_tracking.md)

## Legacy notu

Eski `src/` ve `scripts/` altı final/demo/servis odaklı miras kod içerir. Kaggle için resmi ana yol `src_kaggle/` + `scripts_kaggle/` + `configs/kaggle/war_mode.yaml` hattıdır.
- [Transformer Cross-Encoder Sprint](docs/transformer_cross_encoder_sprint.md)
"# DeeP-PipelinE-teknofest"  
