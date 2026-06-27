# Kaggle Data Contract — TEKNOFEST 2026 E-Ticaret

Bu belge Kaggle War Mode için veri sözleşmesini sabitler. `src_kaggle/` ve `scripts_kaggle/` içinde kolon isimlerinin tek kaynağı `src_kaggle/data/schema.py` dosyasıdır.

## Official input files

### `items.csv`

Zorunlu kolonlar:

- `item_id`
- `title`
- `category`
- `brand`
- `gender`
- `age_group`
- `attributes`

Kontroller:

- `item_id` unique olmalı.
- `title` boş olamaz.
- `gender` ve `age_group` boşsa sistem bunları `unknown` yapar; bu iki alan unknown toleranslıdır.

### `terms.csv`

Zorunlu kolonlar:

- `term_id`
- `query`

Kontroller:

- `term_id` unique olmalı.
- `query` boş olamaz.

### `training_pairs.csv`

Zorunlu kolonlar:

- `id`
- `term_id`
- `item_id`
- `label`

Kontroller:

- `id` unique olmalı.
- `(term_id, item_id)` duplicate olmamalı.
- `label` binary olmalı.
- Resmi train dosyasında `label` sadece `1` olmalı.

### `submission_pairs.csv`

Zorunlu kolonlar:

- `id`
- `term_id`
- `item_id`

Kontroller:

- `id` unique olmalı.
- `(term_id, item_id)` duplicate olmamalı.
- `label` beklenmez.

### `sample_submission.csv`

Zorunlu kolonlar:

- `id`
- `prediction`

## Canonical field map

```yaml
id: id
term_id: term_id
item_id: item_id
query: query
title: title
category: category
brand: brand
gender: gender
age_group: age_group
attributes: attributes
label: label
prediction: prediction
```

Aşağıdaki legacy kolonlar Kaggle çekirdeğinde kullanılmaz:

- `product_name`
- `search_query`
- `product_id`
- `product_color`
- `product_material`

## Generated feature files

`build_pair_dataset.py` iki feature dosyası üretir:

- `data/processed/train_pairs_merged.parquet`
- `data/processed/submission_pairs_merged.parquet`

Train feature kolonları:

- `id`
- `term_id`
- `item_id`
- `label`
- `query`
- `title`
- `category`
- `brand`
- `gender`
- `age_group`
- `attributes`
- `full_item_text`

Submission feature kolonları:

- `id`
- `term_id`
- `item_id`
- `query`
- `title`
- `category`
- `brand`
- `gender`
- `age_group`
- `attributes`
- `full_item_text`

## Validation implementation

Veri sözleşmesi şu dosyalarda uygulanır:

- `src_kaggle/data/schema.py`
- `src_kaggle/data/contracts.py`
- `src_kaggle/data/io.py`
- `src_kaggle/data/pairs.py`
- `src_kaggle/data/pair_builder.py`
- `configs/kaggle/war_mode.yaml`

Hatalar sessizce geçilmez. Eksik kolon, duplicate id, duplicate pair, boş query/title veya yanlış label durumunda `DataContractError` fırlatılır.
