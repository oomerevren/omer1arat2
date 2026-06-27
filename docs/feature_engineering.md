# Feature Engineering Platform

TEKNOFEST Kaggle War Mode için modüler feature platformu `src_kaggle/features/` altında kuruldu. Amaç merged pair dataset'ten CatBoost/LightGBM/Logistic/ensemble modellerine hazır, açıklanabilir feature matrisi üretmektir.

## Komut

```bash
python scripts_kaggle/build_features.py --config configs/kaggle/war_mode.yaml
```

Default çıktılar:

- `data/processed/train_features.parquet`
- `reports/features/feature_catalog.json`
- `reports/features/feature_summary.md`

## Modüller

- `lexical_features.py`
- `category_features.py`
- `brand_features.py`
- `attribute_features.py`
- `gender_age_features.py`
- `query_features.py`
- `retrieval_features.py`
- `semantic_features.py`
- `metadata_features.py`
- `feature_pipeline.py`
- `feature_utils.py`

## Feature grupları

### Lexical

Örnekler:

- `lex_token_overlap_count`
- `lex_token_overlap_ratio`
- `lex_query_coverage_ratio`
- `lex_title_coverage_ratio`
- `lex_char3_jaccard`
- `lex_longest_common_token_span`
- `lex_exact_phrase_match_flag`
- `lex_fuzzy_ratio`
- `lex_token_sort_ratio`
- `lex_token_set_ratio`

### Category

Örnekler:

- `cat_depth`
- `cat_exact_full_match_flag`
- `cat_parent_overlap_flag`
- `cat_leaf_overlap_flag`
- `cat_token_overlap_count`
- `cat_query_coverage_ratio`
- `cat_leaf_query_coverage_ratio`

### Brand

Örnekler:

- `brand_present_flag`
- `brand_query_has_brand_flag`
- `brand_exact_match`
- `brand_token_overlap_count`
- `brand_contradiction_flag`
- `brand_only_query_flag`

### Attribute

Attribute parser ile entegredir.

Örnekler:

- `attr_key_count`
- `attr_value_count`
- `attr_key_overlap_count`
- `attr_value_overlap_count`
- `attr_color_exact_match`
- `attr_color_conflict_flag`
- `attr_material_exact_match`
- `attr_material_conflict_flag`
- `attr_style_overlap`
- `attr_size_numeric_match`
- `attr_conflict_count`

### Gender / age

Örnekler:

- `gender_query_has_cue`
- `age_query_has_cue`
- `gender_exact_match_flag`
- `gender_conflict_flag`
- `age_exact_match_flag`
- `age_conflict_flag`
- `gender_item_unisex_flag`

### Query intent / structure

Query intent modülünden gelen sinyaller `query_` prefix'iyle eklenir.

Örnekler:

- `query_query_length_tokens`
- `query_has_brand_token`
- `query_has_category_token`
- `query_has_color_token`
- `query_is_attribute_heavy`
- `query_possible_typo_or_ambiguous`
- `query_numeric_density`

### Retrieval

Prompt 7 retrieval sistemiyle entegredir.

Örnekler:

- `retrieval_bm25_score`
- `retrieval_dense_score`
- `retrieval_bm25_rank`
- `retrieval_dense_rank`
- `retrieval_in_bm25_topk_flag`
- `retrieval_in_dense_topk_flag`
- `retrieval_rank_agreement_flag`
- `retrieval_lexical_dense_gap`
- `retrieval_same_category_pool_flag`
- `retrieval_same_brand_pool_flag`

### Semantic

CPU-safe TF-IDF cosine tabanlı semantik/sparse similarity sinyalleri:

- `sem_query_title_cosine`
- `sem_query_item_full_text_cosine`
- `sem_query_category_cosine`

### Metadata

Örnekler:

- `meta_title_token_count`
- `meta_query_title_len_ratio`
- `meta_brand_present_flag`
- `meta_attribute_count`
- `meta_missing_field_count`
- `meta_unknown_field_count`
- `meta_title_numeric_density`

## Config

`configs/kaggle/war_mode.yaml`:

```yaml
feature_engineering:
  use_lexical_features: true
  use_category_features: true
  use_brand_features: true
  use_attribute_features: true
  use_gender_age_features: true
  use_query_features: true
  use_retrieval_features: true
  use_semantic_features: true
  use_metadata_features: true
  retrieval_top_k: 100
  semantic_model_name: tfidf_pairwise
  canonical_attribute_keys: [color, material, style, size, pattern, fit, usage]
```

## Null / unknown handling

- Text alanları boş string'e normalize edilir.
- Numeric feature'lar `0.0` ile doldurulur.
- `gender=unknown` ve `age_group=unknown` özel flag'lerle temsil edilir.
- Missing attribute durumunda parser boş dict döndürür.
- Feature matrisi CatBoost/LightGBM için numeric ve null'suz üretilir.

## Feature quality raporları

`write_feature_reports` şunları üretir:

- toplam feature sayısı
- feature group bazlı sayılar
- null oranları
- constant feature listesi
- örnek feature row'ları

## En güçlü olması beklenen gruplar

1. Attribute features
   - color/material/style/size conflict ve exact match sinyalleri özellikle class 0 F1'e yardım eder.
2. Gender / age features
   - `erkek` query + `kadın` item gibi net çelişkiler güçlü negatif sinyaldir.
3. Lexical coverage features
   - exact veya near-exact relevance'i yakalar.
4. Retrieval features
   - BM25/dense rank ve score tabular model için güçlü prior sağlar.
5. Brand contradiction features
   - marka-heavy query'lerde false positive'i azaltır.

## Overfit riski taşıyan feature türleri

- Çok spesifik retrieval rank/score feature'ları, validation split doğru kurulmazsa overfit yapabilir.
- Brand/category raw-frequency etkileri private distribution değişirse riskli olabilir.
- Negative mining kaynaklı `source_pool` türevleri modelin sampler artifact'lerini öğrenmesine neden olabilir; dikkatli validasyon gerekir.
- Dense fallback skorları gerçek semantic modelle değiştirildiğinde distribution shift olabilir.

## Class 0 F1'e özellikle yardım edecekler

- `brand_contradiction_flag`
- `gender_conflict_flag`
- `age_conflict_flag`
- `attr_color_conflict_flag`
- `attr_material_conflict_flag`
- `attr_conflict_count`
- `retrieval_lexical_dense_gap`
- low lexical/query coverage sinyalleri
- same-category / same-brand hard negative metadata feature'ları

## Sonraki entegrasyon

1. `train_with_negatives.parquet` üstünde `build_features.py` çalıştırılmalı.
2. CatBoost/LightGBM baseline bu feature matrisiyle kurulmalı.
3. OOF validation'da feature importance ve segment bazlı F1 raporu çıkarılmalı.
4. Retrieval feature'ları fold-aware hesaplanmalı.
5. Cross-encoder skorları daha sonra tabular ensemble feature'ı olarak eklenmeli.
