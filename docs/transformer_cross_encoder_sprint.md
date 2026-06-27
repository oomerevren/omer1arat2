# Transformer Cross-Encoder Championship Sprint

Bu sprintte cross-encoder hattı iki backend'li net yapıya ayrıldı:

- `sklearn_text`: CPU-safe OOF baseline ve hızlı ablation
- `transformers`: gerçek HuggingFace fine-tuning backend

`backend=transformers` seçildiğinde sessiz fallback yoktur. CUDA yoksa eğitim hata verir; yalnızca debug için `allow_cpu=true` açıkça verilirse CPU denemesi yapılabilir.

## Ana dosyalar

```text
src_kaggle/models/pair_text_builder.py
src_kaggle/models/transformer_dataset.py
src_kaggle/models/transformer_trainer.py
src_kaggle/models/transformer_inference.py
src_kaggle/models/transformer_checkpointing.py
src_kaggle/training/train_cross_encoder.py
scripts_kaggle/train_cross_encoder.py
scripts_kaggle/run_cross_encoder_ablation.py
```

## Text formatları

Desteklenen formatlar:

1. `query_title`
2. `query_title_category`
3. `query_title_category_brand`
4. `full_v1`
5. `full_v2`

`full_v1`:

```text
[QUERY] ... [SEP] [TITLE] ... [SEP] [CATEGORY] ... [SEP] [BRAND] ... [SEP] [ATTR] ... [SEP] [GENDER] ... [SEP] [AGE] ...
```

`full_v2` attribute ve intent bilgisini öne alır:

```text
[QUERY] ... [SEP] [ATTR] ... [SEP] [INTENT] ... [SEP] [TITLE] ... [SEP] [CATEGORY] ... [SEP] [BRAND] ... [SEP] [GENDER] ... [SEP] [AGE] ...
```

## Config

`configs/kaggle/war_mode.yaml` içinde:

```yaml
modeling:
  cross_encoder:
    backend: transformers
    model_name: dbmdz/distilbert-base-turkish-cased
    tokenizer_name: null
    text_format_version: full_v1
    max_length: 256
    batch_size: 16
    eval_batch_size: 32
    learning_rate: 2.0e-5
    weight_decay: 0.01
    warmup_ratio: 0.1
    num_train_epochs: 2
    gradient_accumulation_steps: 1
    fp16: true
    early_stopping_patience: 2
    save_total_limit: 2
    load_best_model_at_end: true
    metric_for_best_model: macro_f1
    class_weighting: none
    allow_cpu: false
```

## Eğitim komutları

Tek deney:

```bash
python scripts_kaggle/train_cross_encoder.py \
  --config configs/kaggle/war_mode.yaml \
  --backend transformers \
  --model-name dbmdz/distilbert-base-turkish-cased \
  --text-format full_v1
```

Experiment engine ile:

```bash
python scripts_kaggle/run_experiment.py \
  --config configs/kaggle/war_mode.yaml \
  --name ce_transformer_full_v1 \
  --model-type cross_encoder
```

Ablation:

```bash
python scripts_kaggle/run_cross_encoder_ablation.py \
  --config configs/kaggle/war_mode.yaml \
  --backbones dbmdz/distilbert-base-turkish-cased,dbmdz/bert-base-turkish-cased \
  --text-formats query_title,query_title_category,full_v1,full_v2 \
  --max-lengths 128,256
```

## OOF ve artefact düzeni

Transformer OOF:

```text
artifacts/oof/cross_encoder_transformer_oof.csv
```

Fold checkpoint yapısı:

```text
artifacts/models/cross_encoder_transformer/<experiment_name>/fold_<k>/
  model/
  tokenizer/
  training_config.json
  metrics.json
  hf_checkpoints/
```

Raporlar:

```text
reports/models/cross_encoder_transformer_cv_report.json
reports/models/cross_encoder_transformer_ablation.csv
reports/models/cross_encoder_transformer_token_stats.json
```

## Token diagnostics

Her fold için validation tarafında:

- average token length
- p95 token length
- max token length
- truncation rate
- max_length
- text_format_version

raporlanır. Bu, `full_v1/full_v2` formatlarında attribute/category bilgisinin kesilip kesilmediğini görmeye yarar.

## Baseline kıyası

Kıyaslanması gerekenler:

- `sklearn_text` OOF macro-F1
- `transformers` OOF macro-F1
- class 0 F1 farkı
- class 1 F1 farkı
- threshold stability
- hard negative segmentleri
- training cost vs gain

Ablation tablosu formatı:

```text
backbone,text_format,max_length,OOF macro-F1,class0 F1,class1 F1,threshold,train_time,note
```

## Şu anki durum

Bu ortamda transformer model indirme/GPU eğitimi çalıştırılmadı. Kod yolu gerçek HF `Trainer` backend'ini kurar ve `backend=transformers` seçildiğinde sessizce `sklearn_text` baseline'a düşmez.

Gerçek yarışma/GPU ortamında ilk önerilen denemeler:

1. `dbmdz/distilbert-base-turkish-cased`, `full_v1`, `max_length=256`
2. `dbmdz/distilbert-base-turkish-cased`, `full_v2`, `max_length=256`
3. `dbmdz/bert-base-turkish-cased`, `full_v1`, `max_length=256`
4. hızlı bütçe varsa `max_length=192` ablation

## Final ensemble önerisi nasıl verilmeli?

Transformer ancak OOF ile kanıtlandıktan sonra final blend'e alınmalı. İlk ağırlık aralığı önerisi:

- transformer güçlü OOF verirse: `0.35–0.55`
- class 0 F1 tabular'dan zayıfsa: `0.20–0.35`
- hard negative segmentlerinde belirgin iyiyse: segment analizi sonrası ağırlık artırılabilir

## Dense retrieval entegrasyonu için kritik noktalar

- Dense retrieval hard negative pool'u transformer eğitimiyle aynı text formatına yakın temsiller kullanmalı.
- Transformer skorları daha sonra tabular feature olarak eklenebilir.
- Pseudo-labeling için transformer + tabular agreement şartı kullanılmalı.
- Dense-nearest hard negatives false-negative safety katmanından geçirilmelidir.
