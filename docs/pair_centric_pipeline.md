# Pair-Centric Merged Dataset Pipeline

TEKNOFEST 2026 Kaggle görevi item classification değildir. Doğru temsil, her satırın bir `(term_id, item_id)` relevance adayı olduğu pair-centric tablodur.

## Inputs

- `data/training_pairs.csv` + `data/terms.csv` + `data/items.csv`
- `data/submission_pairs.csv` + `data/terms.csv` + `data/items.csv`

## Outputs

Default çıktı yolları:

- Train merged dataset: `data/processed/train_pairs_merged.parquet`
- Test merged dataset: `data/processed/submission_pairs_merged.parquet`
- Train kalite raporu: `reports/data_quality/train_pair_build_report.json`
- Test kalite raporu: `reports/data_quality/test_pair_build_report.json`

## Command

```bash
python scripts_kaggle/build_pair_dataset.py --config configs/kaggle/war_mode.yaml
```

CSV istenirse:

```bash
python scripts_kaggle/build_pair_dataset.py \
  --config configs/kaggle/war_mode.yaml \
  --out-train data/processed/train_pairs_merged.csv \
  --out-test data/processed/submission_pairs_merged.csv
```

## Train schema

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

## Test schema

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

## Join quality report

Her split için raporlanan metrikler:

- input pair sayısı
- output row sayısı
- başarılı join satırı
- terms tarafında bulunmayan `term_id` satırı
- items tarafında bulunmayan `item_id` satırı
- null/boş `query` satırı
- null/boş `title` satırı
- duplicate `id` satırı
- duplicate `(term_id, item_id)` satırı
- pair tablosundaki unique term/item sayısı
- missing term/item unique sayısı

Strict mod default açıktır. Join/data quality sorunu varsa çıktı ve rapor yazılır, ardından `DataContractError` ile pipeline durur. Raporu görmek ama hatada durmamak için `--no-strict` kullanılabilir.

## Why this representation is correct

Yarışmada tahmin hedefi ürünün tek başına sınıfı değildir. Aynı item farklı query için relevant veya irrelevant olabilir. Bu nedenle model girdisi her zaman query ile item attribute'larının aynı satırda birleştiği pair-centric tablo olmalıdır. Negative mining, feature engineering, training, validation ve inference aşamalarının tamamı bu merged pair dataset üzerinden ilerlemelidir.

## Design hooks

`full_item_text` alanı şimdiden üretildi. Bu alan ileride:

- retrieval candidate scoring,
- hard-negative mining,
- TF-IDF/BM25 features,
- embedding features,
- cross-encoder input construction

başlıklarında ortak item metni olarak kullanılabilir.
