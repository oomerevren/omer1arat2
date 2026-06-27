# Attributes Parser and Normalization

TEKNOFEST Kaggle verisinde item attribute bilgileri tek kolonda gelir:

```text
attributes = "anahtar: değer, anahtar: değer, ..."
```

Bu alan artık `src_kaggle/data/attribute_parser.py` ile parse edilir ve modellemeye hazır kolonlara çevrilir.

## Üretilen alanlar

- `attribute_dict`
  - canonical key -> normalized value list
  - CSV/parquet stabilitesi için JSON string olarak saklanır
- `attribute_keys`
  - pipe-separated key list
- `attribute_values`
  - pipe-separated normalized value list
- `normalized_attribute_text`
  - deterministic key-value text
- `color_value`
- `material_value`
- `style_value`

## Canonical key seti

Başlıca key alias kuralları:

| Raw key örnekleri | Canonical key |
|---|---|
| `renk`, `ürün rengi`, `color`, `colour` | `color` |
| `materyal`, `malzeme`, `kumaş tipi`, `fabric`, `material` | `material` |
| `stil`, `style`, `tarz`, `model` | `style` |
| `desen`, `pattern` | `pattern` |
| `beden`, `size` | `size` |
| `kalıp`, `fit` | `fit` |
| `topuk boyu`, `heel height` | `heel_height` |

Bilinmeyen key'ler atılmaz; ASCII-normalized slug formuna çevrilir.

## Value normalization

Renk örnekleri:

- `siyah`, `black` -> `black`
- `beyaz`, `white` -> `white`
- `bej`, `beige` -> `beige`
- `kırmızı`, `kirmizi`, `red` -> `red`

Materyal örnekleri:

- `deri` -> `leather`
- `hakiki deri`, `gerçek deri`, `genuine leather` -> `genuine_leather`
- `suni deri`, `sentetik deri`, `faux leather` -> `faux_leather`
- `pamuk`, `cotton` -> `cotton`

Önemli: `leather`, `genuine_leather` ve `faux_leather` aynılaştırılmaz. Anlamsal fark korunur.

## Edge case desteği

Desteklenen durumlar:

- boş string
- null / NaN
- malformed fragment
- tekrar eden key
- aynı key için çoklu value: `/`, `|`, `;`, `+`, `ve`, `&`
- büyük/küçük harf farkı
- Türkçe karakter farkları: `kırmızı` / `kirmizi`, `ürün rengi` / `urun rengi`

Malformed fragment örneği:

```text
"renk siyah, materyal: deri"
```

Çıktı:

```json
{
  "material": ["leather"],
  "unknown": ["renk_siyah"]
}
```

Bilgi kaybı olmaması için bozuk fragment `unknown` altında korunur.

## Örnek input/output

Input:

```text
renk: siyah, materyal: hakiki deri, stil: klasik
```

Output:

```json
{
  "attribute_dict": {"color": ["black"], "material": ["genuine_leather"], "style": ["classic"]},
  "attribute_keys": "color|material|style",
  "attribute_values": "black|genuine_leather|classic",
  "normalized_attribute_text": "color: black ; material: genuine_leather ; style: classic",
  "color_value": "black",
  "material_value": "genuine_leather",
  "style_value": "classic"
}
```

## Pipeline entegrasyonu

`pair_builder.py` içinde item tablosu merge edilmeden önce attribute feature kolonları üretilir. Böylece merged dataset şu alanları içerir:

- raw `attributes`
- parsed `attribute_dict`
- normalized `normalized_attribute_text`
- shortcut değerler: `color_value`, `material_value`, `style_value`
- `full_item_text`

`full_item_text` artık raw attributes yerine `normalized_attribute_text` kullanır.

## Açık kalan noktalar

- Alias sözlüğü yarışma verisinin gerçek dağılımı görüldükçe genişletilmeli.
- Değer normalizasyonunda long-tail kategori/materyal adları için frekans analizi yapılmalı.
- Çok karmaşık attribute string'lerinde virgül value içinde geçerse basit parser fragment split yapabilir; bilgi `unknown` altında korunur ama ileride daha gelişmiş parser eklenebilir.
