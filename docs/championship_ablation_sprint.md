# Championship Ablation Sprint

Bu sprintin amacı yeni bileşen eklemek değil, Kaggle War Mode pipeline'ında hangi parçanın gerçekten tutulacağını OOF-first disiplinle kanıtlamaktır.

## Eklenen ablation sistemi

Yeni modüller:

```text
src_kaggle/ablation/
  ablation_specs.py
  component_toggle.py
  ablation_runner.py
  ablation_reporting.py
  ablation_registry.py
  risk_assessment.py
```

Yeni CLI'lar:

```bash
python scripts_kaggle/run_ablation_suite.py --config configs/kaggle/war_mode.yaml
python scripts_kaggle/summarize_ablation_results.py --master reports/ablation/ablation_master_table.csv
```

## Koşulacak ablation aileleri

Sistem 59 kontrollü ablation spec'i üretir:

- feature group ablation
- negative mining ablation
- retrieval ablation
- special signal ablation
- model family ablation
- threshold ablation
- dense semantic ablation

Her spec tek anlamlı component değişikliği içerir ve merkezi tabloya yazılır.

## Üretilen raporlar

```text
reports/ablation/ablation_master_table.csv
reports/ablation/ablation_summary.md
reports/ablation/feature_group_ablation.csv
reports/ablation/negative_mining_ablation.csv
reports/ablation/model_family_ablation.csv
reports/ablation/dense_ablation.csv
reports/ablation/threshold_ablation.csv
reports/ablation/retrieval_ablation.csv
reports/ablation/risk_flags.csv
reports/ablation/final_keep_risky_drop.csv
reports/ablation/final_pipeline_recommendation.md
```

## OOF-first karar kuralı

Final kararlar şu sinyallerle verilir:

- OOF macro-F1
- class0 F1
- class1 F1
- best threshold
- threshold fragility
- seed stability / seed std
- segment raporları
- public LB varsa yalnızca not olarak public–OOF delta

Public LB tek başına seçim kriteri değildir.

## Risk flag sistemi

Her ablation sonucu şu risk etiketlerinden birini alır:

- `low_risk_keep`
- `medium_risk_monitor`
- `high_risk_public_only`
- `artifact_suspected`
- `needs_fold_safe_recheck`

Dense/retrieval/negative mining gibi label veya retrieval artefact riski taşıyan bileşenler OOF lift verse bile fold-safe recheck gerektirebilir.

## Keep / risky / drop sistemi

Her bileşen şu sınıflardan birine düşer:

- Keep no matter what
- Good but risky
- Segment benefit only
- Drop from final pipeline

Bu karar `reports/ablation/final_keep_risky_drop.csv` ve `final_pipeline_recommendation.md` içine yazılır.

## Bu workspace'teki durum

Official Kaggle dosyaları bu workspace'te bulunmadığı için gerçek OOF skorları üretilmedi. Sistem skor uydurmak yerine tüm ablation spec'lerini `not_run` statüsü ve açık `missing_data` notu ile master table'a yazdı.

Official data yerleştirilince çalıştırma sırası:

```bash
python scripts_kaggle/build_pair_dataset.py --config configs/kaggle/war_mode.yaml
python scripts_kaggle/build_retrieval_index.py --config configs/kaggle/war_mode.yaml --mode hybrid
python scripts_kaggle/build_negatives.py --config configs/kaggle/war_mode.yaml --use-dense true
python scripts_kaggle/build_features.py --config configs/kaggle/war_mode.yaml
python scripts_kaggle/run_ablation_suite.py --config configs/kaggle/war_mode.yaml
```

Transformer ve real dense içeren deneyler için explicit izin gerekir:

```bash
python scripts_kaggle/run_ablation_suite.py \
  --config configs/kaggle/war_mode.yaml \
  --allow-transformer \
  --allow-real-dense
```

## Dense retrieval gerçekten işe yaradı mı?

Bu sprintte karar mekanizması kuruldu; official data olmadığı için gerçek skor iddiası yapılmadı. Dense'in işe yarayıp yaramadığı şu deneylerle kanıtlanacak:

- `dense_no_dense_anywhere`
- `dense_features_only`
- `dense_hard_negatives_only`
- `dense_features_plus_dense_negatives`
- `dense_text_v1`
- `dense_text_v2`

Dense finalde ana bileşen değil, OOF kanıtı gelene kadar yardımcı semantic sinyal ve hard-negative kaynağıdır.

## Transformer CE gerçekten fark yarattı mı?

Gerçek transformer CE bu ortamda çalıştırılmadı; sessiz fallback yoktur. CE farkı şu deneylerle ölçülecek:

- `sklearn_text_baseline`
- `transformer_ce_best`
- `tabular_plus_transformer`
- `tabular_plus_transformer_plus_dense_signals`

Transformer, ancak OOF'ta tabular'a tamamlayıcı lift ve class0/class1 stabilitesi gösterirse final blend'in ana parçası olur.

## Final sadeleşme prensibi

Final pipeline'a yalnızca şu koşulları sağlayan parçalar kalır:

1. OOF macro-F1 veya class0 F1 katkısı var.
2. Threshold kırılganlığı düşük.
3. Segmentte değerliyse hangi segmentte değerli olduğu raporlu.
4. Retrieval/negative mining için fold-safe risk notu temiz.
5. Public LB artışı OOF düşüşüyle birlikte gelmiyor.
