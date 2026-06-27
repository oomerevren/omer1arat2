# Dense Retrieval ve Semantic Hard Negative Sprinti

Bu sprint Kaggle War Mode hattında lexical/BM25 omurgasına gerçek semantic retrieval katmanı ekler ve bu sinyali negative mining ile feature engineering'e taşır.

## Dense backend

- Ana backend: `retrieval.dense.backend: real_dense`
- Varsayılan model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Debug backend: `fallback_dense` yalnızca açık seçilirse çalışır.
- `real_dense` seçiliyken dependency/model yükleme hatasında sessiz fallback yoktur; açık `RuntimeError` verilir.

## Item/query text formatları

- `dense_v1`: title + category + brand + normalized attributes + gender + age_group dengeli ürün kartı.
- `dense_v2`: title/leaf-category/canonical attribute öncelikli, daha az tekrar ve daha az gürültü.
- Query tarafı başlangıçta `raw`/normalized güvenli temsil kullanır; aşırı intent şişirmesi yapılmaz.

## Persist ve versiyonlama

Dense artefact düzeni:

```text
artifacts/retrieval/dense/<model_name>/<item_text_version>/<config_hash>/
  item_embeddings.npy
  item_ids.csv
  metadata.json
  index.pkl
```

Metadata model name, backend, item text version, item count, embedding dim, timestamp, config hash ve `semantic_backend_active` alanlarını taşır.

## BM25'e göre kazanım

`evaluate_dense_retrieval.py` şu soruları raporlar:

- BM25/dense overlap oranı
- dense-only aday sayısı
- segment bazlı davranış: short query, brand-heavy, attribute-heavy, gender cue, long-tail
- dense top-k category/brand match oranları

Çıktılar:

```text
reports/retrieval/dense_vs_bm25_comparison.csv
reports/retrieval/query_segment_retrieval_comparison.json
reports/retrieval/semantic_confuser_examples.md
```

## Semantic hard negatives

Akış:

1. Her term için bilinen positives dışlanır.
2. Dense top-k aday getirilir.
3. Lexical/category/brand/attribute/conflict metadata eklenir.
4. False-negative safety filter uygulanır.
5. Güvenli aday training'e, riskli aday uncertain/skip havuzuna gider.

Alt tipler:

- `dense_same_category_hard`
- `dense_same_brand_hard`
- `dense_semantic_confuser`
- `dense_attribute_near_miss`
- `dense_hard`

## False negative risk yönetimi

Risk artıran koşullar:

- very high dense score
- high dense + high lexical
- same brand + same category + conflict yok
- semantic variant riski

Güvenli negatif sinyalleri:

- gender/age/color/material conflict
- dense yakın ama lexical coverage zayıf
- category sibling farkı

Status alanları:

- `safe_negative`
- `hard_negative`
- `uncertain_candidate`
- `skipped_due_to_false_negative_risk`

## Feature engineering'e taşınan sinyaller

Eklenen önemli feature'lar:

- `retrieval_dense_score_real`
- `retrieval_dense_rank_real`
- `retrieval_dense_rank_percentile`
- `retrieval_dense_only_hit_flag`
- `retrieval_bm25_dense_overlap_flag`
- `retrieval_dense_bm25_rank_gap`
- `retrieval_hybrid_consensus_flag`
- `retrieval_low_lexical_high_dense_flag`
- `retrieval_high_lexical_low_dense_flag`
- `retrieval_category_match_dense_high_flag`
- `retrieval_attribute_conflict_dense_high_flag`

Class 0 F1 potansiyeli özellikle dense-high + attribute/gender conflict ve dense-only semantic confuser örneklerinden gelir.

## Leakage / OOF planı

- Item universe public auxiliary olduğu için item embedding index global kurulabilir.
- Ancak hard negative selection sırasında positive exclusion ve candidate seçim raporları fold train positives ile yeniden üretilmelidir.
- OOF için `build_negatives.py --mode fold_aware --active-fold k --use-dense true` pattern'i kullanılmalıdır.
- Dense feature'lar label kullanmaz; fakat negative mining label-aware olduğundan fold-aware yürütülmezse validation iyimserleşebilir.

## Final validation / ensemble önerisi

- Dense features tabular modelde class-0 ayrımı için güçlüdür.
- Cross-encoder tarafında semantic hard negatives modelin karar sınırını zorlaştırır.
- Final blend'de dense sinyaller hem tabular hem CE tarafına fayda sağlar; en büyük doğrudan katkı tabular feature setinde, en büyük dolaylı katkı CE hard-negative eğitimindedir.

## Mutlaka test edilecek ablation varyantları

1. `dense_v1` vs `dense_v2`
2. `real_dense` model A/B: multilingual MiniLM vs Türkçe sentence model
3. dense negatives per positive: 1/2/4
4. false-negative thresholds sıkı vs gevşek
5. BM25-only vs dense-only vs hybrid negative mining
