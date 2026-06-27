# Retrieval / Candidate Generation System

TEKNOFEST Kaggle War Mode için retrieval katmanı `src_kaggle/retrieval/` altında kuruldu. Amaç yalnızca top-k arama değil; hard negative mining, pseudo-labeling, error analysis ve feature engineering için açıklanabilir candidate pool üretmektir.

## Modüller

- `item_text_builder.py`
  - `retrieval_text`, `dense_text`, `full_item_text` üretir.
- `bm25_retriever.py`
  - lightweight BM25 index, top-k scoring, persist/reload.
- `dense_retriever.py`
  - dense API; sentence-transformers varsa kullanır, yoksa deterministic TF-IDF + SVD fallback kullanır.
- `hybrid_retriever.py`
  - BM25, dense, category, brand, random-far, attribute-similar havuzlarını tek arayüzde toplar.
- `retrieval_index.py`
  - build/load/save yardımcıları.
- `scripts_kaggle/build_retrieval_index.py`
  - index build ve rapor CLI'ı.

## Komut

```bash
python scripts_kaggle/build_retrieval_index.py --config configs/kaggle/war_mode.yaml
```

Default çıktılar:

- `artifacts/retrieval/hybrid_retriever.pkl`
- `reports/retrieval/index_report.json`
- `reports/retrieval/retrieval_examples.md`

## Item representation

Her item için üç temsil hedeflendi:

- `full_item_text`
- `retrieval_text`
- `dense_text`

Alanlar:

- `title`
- `category`
- `brand`
- `gender`
- `age_group`
- `normalized_attribute_text`
- `color_value`
- `material_value`
- `style_value`

BM25 için daha token-dense `retrieval_text`, dense encoder için etiketli/semantik `dense_text` kullanılır.

## API örnekleri

```python
from src_kaggle.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever.build(items, dense_model_name="tfidf_svd_fallback")

bm25 = retriever.lexical_nearest_pool("nike beyaz kadın sneaker", top_k=50)
dense = retriever.dense_nearest_pool("erkek deri ceket", top_k=50)
hybrid = retriever.hybrid_search("siyah oversize sweatshirt", top_k=50)
cat = retriever.same_category_pool({"ceket"}, top_k=30)
brand = retriever.same_brand_pool({"Nike"}, top_k=20)
far = retriever.random_far_pool("nike beyaz kadın sneaker", top_k=30)
attr = retriever.attribute_similar_pool("siyah deri ceket", top_k=30)
```

Her sonuçta açıklanabilir kolonlar bulunur:

- `item_id`
- `title`
- `category`
- `brand`
- `source`
- `score`
- retriever'a göre `bm25_score`, `dense_score` veya `hybrid_score`

## Config

`configs/kaggle/war_mode.yaml`:

```yaml
retrieval:
  enabled: true
  seed: 42
  enabled_retrievers: [bm25, dense]
  bm25_top_k: 50
  dense_top_k: 50
  same_category_top_k: 30
  same_brand_top_k: 20
  random_far_top_k: 30
  dense_model_name: tfidf_svd_fallback
  use_full_item_text: true
  use_attribute_text: true
  persist_indices: true
  embedding_cache_dir: artifacts/retrieval/embeddings
  index_path: artifacts/retrieval/hybrid_retriever.pkl
```

## Hangi bileşen nerede güçlü?

### BM25 / lexical

Güçlü olduğu query segmentleri:

- brand-heavy query: `nike beyaz kadın sneaker`
- exact ürün tipi geçen query
- renk/materyal/category tokenları doğrudan geçen query
- kısa ve net query

Zayıf olduğu durumlar:

- lexical olarak farklı ama semantik olarak yakın ürünler
- typo yoğun query
- synonym/varyant kelimeler

### Dense retrieval

Güçlü olduğu durumlar:

- query ile title aynı kelimeleri paylaşmıyor ama intent yakın
- semantic hard negative arama
- pseudo-labeling için benzer item keşfi
- long-tail query varyasyonları

Not: Bu aşamada default dense `tfidf_svd_fallback` ile CPU-safe çalışır. `sentence-transformers` modeli config'ten verildiğinde gerçek embedding encoder denenir.

### Category-aware pools

Güçlü olduğu durumlar:

- hard negative mining
- aynı kategori içinde yanlış alt ürün/intent ayrımı
- validation hata analizi

### Same-brand pools

Güçlü olduğu durumlar:

- modelin “marka gördüm relevant” shortcut'ını kırmak
- brand-heavy query segmentleri

### Random-far pool

Güçlü olduğu durumlar:

- temel `label=0` sınıfını öğretmek
- easy negative dengesi kurmak

### Attribute-similar pool

Güçlü olduğu durumlar:

- renk/materyal/stil sinyali içeren attribute-heavy query
- hard negative / pseudo-positive aday keşfi

## Negative mining entegrasyon notu

Mevcut `negative_pools.py` kendi TF-IDF pool yapısını kullanıyor; yeni `HybridRetriever` API aynı havuzları daha geniş şekilde sunar. Sonraki adımda negative mining şu fonksiyonları doğrudan kullanacak şekilde bağlanmalı:

- `lexical_nearest_pool`
- `dense_nearest_pool`
- `same_category_pool`
- `same_brand_pool`
- `random_far_pool`
- `attribute_similar_pool`

Bu sayede lexical/dense skorlar aynı anda negative metadata ve feature engineering girdisi olur.

## Raporlar

`build_retrieval_index.py` şu raporları üretir:

- toplam item sayısı
- item text boşluk oranı
- index başarı durumu
- enabled retriever listesi
- örnek query bazında BM25 top-k
- dense top-k
- hybrid top-k
- BM25/dense overlap Jaccard

Örnek query'ler:

- `nike beyaz kadın sneaker`
- `erkek deri ceket`
- `bebek zıbın`
- `unisex okul çantası`
- `42 numara koşu ayakkabısı`
- `siyah oversize sweatshirt`

## Sonraki entegrasyon

1. Negative mining `HybridRetriever` ile refactor edilmeli.
2. Retrieval score'ları pair feature olarak eklenmeli: BM25, dense cosine, hybrid rank, source overlap.
3. Segment bazlı retrieval recall/coverage raporu eklenmeli.
4. Dense encoder için yarışma ortamında hızlı ve güçlü Türkçe/multilingual model seçilmeli.
5. Validation pipeline retrieval pool kalitesini macro-F1 etkisiyle ölçmeli.
