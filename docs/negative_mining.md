# Hard Negative Mining — Kaggle War Mode

`src_kaggle/data/negative_mining.py` TEKNOFEST 2026 Kaggle için multi-layer, açıklanabilir, false-negative kontrollü negative mining çekirdeğidir.

## Komut

Önce pair-centric merged train üret:

```bash
python scripts_kaggle/build_pair_dataset.py --config configs/kaggle/war_mode.yaml
```

Sonra negatifleri üret:

```bash
python scripts_kaggle/build_negatives.py --config configs/kaggle/war_mode.yaml
```

Default çıktılar:

- augmented train: `data/processed/train_with_negatives.parquet`
- uncertain adaylar: `data/processed/uncertain_negative_candidates.parquet`
- JSON rapor: `reports/negative_mining/negative_mining_report.json`
- Markdown rapor: `reports/negative_mining/negative_mining_report.md`

## Modüller

- `src_kaggle/data/negative_pools.py`
  - candidate pool üretimi
- `src_kaggle/data/negative_filters.py`
  - false-negative safety layer
- `src_kaggle/data/negative_mining.py`
  - sampling orkestrasyonu, augmented dataset, rapor
- `scripts_kaggle/build_negatives.py`
  - CLI entrypoint

## Negative türleri

Desteklenen türler:

1. `easy`
   - random far pool
   - düşük lexical overlap
   - temel `label=0` öğrenimi için

2. `same_category`
   - pozitif item kategorisiyle aynı kategori
   - yanlış ürün/alt intent yakalamak için

3. `same_brand`
   - pozitif marka ile aynı marka
   - modelin marka shortcut'ına düşmesini önlemek için

4. `lexical_confusing`
   - TF-IDF cosine/BM25-benzeri lexical nearest pool
   - kelime örtüşmesine aşırı bağımlılığı kırmak için

5. `attribute_conflict`
   - color/material/gender/age gibi conflict sinyalleri
   - attribute mismatch öğretmek için

6. `dense_hard`
   - config alanı hazır
   - şu an dense retriever eklenene kadar lexical hard proxy olarak tasarlandı; default kapalı

## False-negative safety layer

Her aday şu kontrollerden geçer:

- aynı term/query için bilinen pozitif item mı?
- lexical similarity çok yüksek mi?
- same category + same brand + yüksek lexical skor potansiyel varyant mı?
- dense similarity çok yüksek mi? Şu an placeholder, dense retriever eklenince aktif kullanılacak.
- color/material/gender/age conflict var mı?

Adaylar şu statülerden biriyle işaretlenir:

- `safe_negative`
- `hard_negative`
- `uncertain_candidate`

`exclude_uncertain: true` ise uncertain adaylar eğitime alınmaz, ayrı dosyaya yazılır.

## Output kolonları

Negatif dataset metadata içerir:

- `term_id`
- `item_id`
- `query`
- `title`
- `label = 0`
- `negative_type`
- `source_pool`
- `safety_status`
- `safety_score`
- `hardness_score`
- `lexical_score`
- `dense_score`
- `category_match_flag`
- `brand_match_flag`
- `gender_conflict_flag`
- `age_conflict_flag`
- `color_conflict_flag`
- `material_conflict_flag`
- `safety_reasons`

Augmented train dosyasında official pozitifler de korunur ve `negative_type=positive` olarak işaretlenir.

## Config

`configs/kaggle/war_mode.yaml` içinde:

```yaml
negative_mining:
  enabled: true
  seed: 42
  mode: global
  fold_column: fold
  active_fold: null
  exclude_uncertain: true
  max_candidates_per_query: 250
  easy_negatives_per_positive: 1
  same_category_negatives_per_positive: 1
  same_brand_negatives_per_positive: 1
  lexical_negatives_per_positive: 1
  attribute_conflict_negatives_per_positive: 1
  dense_negatives_per_positive: 0
  use_bm25_pool: true
  use_dense_pool: false
  use_same_category_pool: true
  use_same_brand_pool: true
  false_negative_thresholds:
    very_high_lexical: 0.92
    variant_lexical: 0.82
    hard_lexical: 0.35
    very_high_dense: 0.94
```

Aynı seed ile deterministic sampling hedeflenir.

## Fold-aware tasarım

`NegativeMiningConfig.mode = fold_aware` ve `active_fold` verildiğinde sistem sadece training fold satırlarını kullanarak mining yapar:

```bash
python scripts_kaggle/build_negatives.py \
  --config configs/kaggle/war_mode.yaml \
  --mode fold_aware \
  --active-fold 0
```

Beklenen kullanım:

1. merged train üzerinde fold kolonu oluşturulur.
2. validation fold seçilir.
3. negative mining sadece train fold pozitiflerinden candidate üretir.
4. validation fold tarafı mining kararlarına kaynak olmaz.

Bu, validation leakage riskini azaltmak için tasarlanmıştır. Sonraki validation pipeline bu modu doğrudan kullanmalıdır.

## Örnek davranış

Sentetik testte örnekler:

- `nike beyaz kadın sneaker`
  - `same_brand`: `Nike erkek şort`
  - `lexical_confusing`: `Adidas beyaz spor çorap`
  - pozitif item tekrar negatif yapılmadı.

- `erkek deri ceket`
  - `same_category`: `Erkek suni deri ceket`
  - yüksek lexical + category/brand sinyali nedeniyle `hard_negative`

- `bebek zıbın`
  - easy/lexical negatifler üretildi; pozitif bebek zıbın dışlandı.

- `unisex okul çantası`
  - random far ve lexical pool'dan negatifler üretildi.

- `42 numara koşu ayakkabısı`
  - size cue query intent tarafında işaretlenir; negatifler metadata ile üretilir.

Çalıştırılabilir örnek:

```bash
python tests_kaggle/test_negative_mining_examples.py
```

## Raporlanan metrikler

- toplam negatif sayısı
- query başına ortalama negatif
- negative type dağılımı
- source pool dağılımı
- same-category oranı
- same-brand oranı
- uncertain oranı
- lexical score dağılımı
- hardness score dağılımı
- category bazlı negatif yoğunluğu
- problem query segmentleri

## Güçlü yanlar

- Tamamen random sampler değil; multi-layer pool sistemi var.
- False-negative safety layer açık ve metadata üretiyor.
- Query intent + attribute parser ile entegre.
- Deterministik seed yapısı var.
- Fold-aware moda hazır.
- Üretilen örnekler black-box değil; `negative_type`, `source_pool`, `safety_reasons` ile izlenebilir.

## Kalan riskler

- Dense hard negative henüz gerçek embedding retriever kullanmıyor; lexical proxy var.
- False-negative safety heuristic; gerçek label olmadığı için kusursuz olamaz.
- Category hierarchy yok; `category` raw string eşleşmesi kullanılıyor.
- Çok büyük item evreninde TF-IDF cosine top-k daha optimize hale getirilmeli.
- Same-category/same-brand pool küçükse hedeflenen negatif sayısına ulaşılamayabilir.

## Sonraki entegrasyon adımları

1. Validation pipeline fold-aware mining ile bağlanmalı.
2. Dense retriever / embedding index eklenmeli.
3. Segment bazlı OOF performansına göre negative mix oranları optimize edilmeli.
4. Uncertain adaylar ayrı analiz edilip pseudo-label/teacher model ile filtrelenmeli.
5. Category hierarchy ve brand phrase matching güçlendirilmeli.
