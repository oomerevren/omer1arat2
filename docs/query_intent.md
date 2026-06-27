# Query Intent / Segment Extraction

`src_kaggle/features/query_intent.py` rule-based ve açıklanabilir query segmentasyon motorudur. Amaç tüm query'lere aynı davranmak yerine feature engineering, validation, threshold tuning, hata analizi ve negative sampling için yapısal sinyaller üretmektir.

## Üretilen sinyaller

Her `query` için başlıca kolonlar:

- `query_normalized`
- `query_length_tokens`
- `query_char_length`
- `has_brand_token`
- `has_category_token`
- `has_color_token`
- `has_material_token`
- `has_style_token`
- `has_gender_token`
- `has_age_token`
- `has_size_token`
- `is_short_query`
- `is_long_query`
- `is_attribute_heavy`
- `is_brand_heavy`
- `is_category_heavy`
- `possible_typo_or_ambiguous`
- `detected_brand_candidates`
- `detected_category_candidates`
- `detected_color_candidates`
- `detected_material_candidates`
- `detected_style_candidates`
- `detected_gender_candidates`
- `detected_age_candidates`

Boolean sinyaller modellemeye uygun olması için `0/1` integer olarak üretilir. Candidate listeleri `|` ile birleştirilmiş string olarak saklanır.

## Veri kaynakları

Motor şu kaynaklardan sözlük üretir:

- query text
- `items.brand` marka sözlüğü
- `items.category` kategori tokenları
- `items.attributes` içinden parse edilen color/material/style değerleri
- sabit gender/age/size kelime listeleri
- attribute parser alias sözlükleri

## Kural seti

### Brand

`items.brand` değerlerinden token ve phrase sözlüğü çıkarılır. Query içinde marka tokenı/phrase'i geçerse:

- `has_brand_token = 1`
- `detected_brand_candidates` dolar
- query kısa ve marka içeriyorsa `is_brand_heavy = 1`

### Category

Kategori sinyali iki kaynaktan gelir:

- base category words: `sneaker`, `ayakkabı`, `ceket`, `çanta`, `zıbın`, `sweatshirt`, vb.
- `items.category` tokenları

### Attribute

Renk, materyal ve stil sinyalleri `attribute_parser.py` alias sözlükleriyle uyumludur.

Örnek:

- `siyah`, `black` -> color
- `deri`, `hakiki deri`, `suni deri` -> material sinyali
- `oversize`, `klasik`, `spor` -> style sinyali

`is_attribute_heavy = 1` için:

- color/material/style sinyal sayısı en az 2 ise veya
- query tokenlarının en az yarısı attribute sinyali ise

### Gender

Örnek mapping:

- `kadın`, `bayan` -> `female`
- `erkek`, `bay` -> `male`
- `kız` -> `female_child`
- `unisex` -> `unisex`

### Age

Örnek mapping:

- `bebek` -> `baby`
- `yenidoğan` -> `newborn`
- `çocuk` -> `child`
- `genç` -> `teen`
- `okul` -> `school_age`

### Size / number

Şu sinyaller size kabul edilir:

- `numara`, `beden`, `yaş`, `ay`, `cm`, `kg`
- numeric tokenlar: `42`, `38`, `110`
- alpha size tokenları: `xs`, `s`, `m`, `l`, `xl`, `xxl`

### Typo / ambiguous heuristic

`possible_typo_or_ambiguous = 1` şu durumlarda tetiklenebilir:

- boş query
- aşırı tekrar karakter: `siyahhhh`
- çok uzun tek token
- uzun ve sesli harfsiz token
- çok kısa query ve bilinen sözlüklerde hiç eşleşme yok

Bu sinyal kesin typo detektörü değildir; hata analizi ve segment bazlı threshold incelemesi için heuristik bayraktır.

## Örnek input-output

| query | brand | color | material | gender | age | size | category | segment |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `nike beyaz kadın sneaker` | 1 | 1 | 0 | 1 | 0 | 0 | 1 | brand + color + gender + category |
| `bebek zıbın` | 0 | 0 | 0 | 0 | 1 | 0 | 1 | short age/category |
| `erkek deri ceket` | 0 | 0 | 1 | 1 | 0 | 0 | 1 | material + gender + category |
| `unisex okul çantası` | 0 | 0 | 0 | 1 | 1 | 0 | 1 | gender + school-age + category |
| `42 numara koşu ayakkabısı` | 0 | 0 | 0 | 0 | 0 | 1 | 1 | size + category |
| `siyah oversize sweatshirt` | 0 | 1 | 0 | 0 | 0 | 0 | 1 | attribute-heavy |

## Pipeline entegrasyonu

`pair_builder.py` merged train/test üretirken query intent kolonlarını da ekler. Böylece downstream süreçler aynı merged dataset üzerinden segment bazlı çalışabilir:

- feature engineering
- error analysis
- segment bazlı validation
- threshold tuning
- negative mining stratejisi

## Güçlü sinyaller vs heuristik sinyaller

Daha güçlü sinyaller:

- brand match: item brand sözlüğüne dayanır
- category match: item category + base kategori sözlüğü
- color/material/style: attribute parser sözlükleriyle uyumlu
- gender/age: küçük ama açıklanabilir kelime listeleri
- size: numeric/beden cue kuralları

Heuristik sinyaller:

- `possible_typo_or_ambiguous`
- `is_brand_heavy`
- `is_attribute_heavy`
- `is_category_heavy`

Bu heuristikler skor optimizasyonundan önce segment analiziyle doğrulanmalıdır.

## Gelecek geliştirmeler

- Gerçek train/test query frekanslarına göre sözlük genişletme
- Brand phrase matching için daha güçlü n-gram eşleme
- Category hierarchy parsing
- Typo detection için edit-distance / SymSpell tabanlı aday düzeltme
- Segment bazlı OOF metric dashboard
- Segment bazlı threshold tuning
- Query intent'e göre negative mining politikası
