# Competition Freeze Guide — TEKNOFEST 2026 Kaggle

Bu doküman geliştirme dökümanı değil, yarışma günü operasyon kılavuzudur.

## 1. Final repo yapısı

| Zone | Path | Status | Use |
| --- | --- | --- | --- |
| Final | `src_kaggle/`, `scripts_kaggle/`, `configs/kaggle/final/`, `artifacts/final/`, `reports/final/` | frozen | yarışma operasyonu |
| Experimental | `configs/kaggle/experiments/`, `reports/ablation/`, `reports/leaderboard/` | mutable | ablation / analiz |
| Legacy | `src/`, `scripts/`, `src_hackathon/`, `scripts_hackathon/` | non-final | referans / hackathon |

Final submission günü legacy `scripts/` kullanılmaz.

## 2. Final config’ler

```text
configs/kaggle/final/shared_final_base.yaml
configs/kaggle/final/family_A_balanced.yaml
configs/kaggle/final/family_B_defensive.yaml
configs/kaggle/final/family_C_aggressive.yaml
```

Her family config şunları içermelidir:

- `final_mode: true`
- `frozen: true`
- family name
- models used
- ensemble method / weights
- threshold
- source experiment ids
- data / negative mining / retrieval / feature / validation version
- artefact paths
- submission output path

## 3. Family A/B/C nasıl yeniden üretilir?

Önce leaderboard ve final family raporları güncellenir:

```bash
python scripts_kaggle/analyze_leaderboard_correlation.py
python scripts_kaggle/build_final_submission_families.py --config configs/kaggle/war_mode.yaml
```

Sonra final package/freeze çalıştırılır:

```bash
python scripts_kaggle/package_final_families.py
python scripts_kaggle/validate_final_release.py
```

Official test prediction dosyaları ve `data/submission_pairs.csv` hazırsa family submission dosyaları validator’dan geçerek üretilir.

## 4. Final artefact standardı

```text
artifacts/final/
  models/
  families/
    family_A_balanced/
      metadata.json
      selected_models.json
      threshold.json
      submission_preview.csv
    family_B_defensive/
    family_C_aggressive/
  submissions/
    family_A_balanced_submission.csv
    family_B_defensive_submission.csv
    family_C_aggressive_submission.csv
  manifests/
    final_artifact_manifest.json
```

Ana manifest:

```text
reports/final/final_artifact_manifest.json
```

## 5. Submission validator

Final submission ancak şu koşulla upload edilir:

- `submission.csv` mevcut
- `submission_validation_report.json` mevcut
- kolonlar yalnızca `id,prediction`
- id sırası / seti official `submission_pairs.csv` ile uyumlu
- prediction binary
- registry kaydı mevcut

## 6. Yarışma günü sıra

1. Family seç: A / B / C.
2. `configs/kaggle/final/<family>.yaml` aç, `final_mode: true` kontrol et.
3. `reports/final/final_artifact_manifest.json` kontrol et.
4. `python scripts_kaggle/validate_final_release.py` çalıştır.
5. Submission validator raporunu oku.
6. `reports/submissions/submission_registry.csv` içinde kayıt var mı kontrol et.
7. Public LB sonucu manuel olarak `analyze_leaderboard_correlation.py --add-public-entry ...` ile işle.
8. Decision table’a göre family değişimi yap veya Family A’yı koru.

## 7. Freeze sonrası değişiklik disiplini

- Final config üzerinde doğrudan değişiklik yapılmaz.
- Yeni deneme gerekiyorsa `configs/kaggle/experiments/<name>.yaml` açılır.
- Deneme OOF + class0 + risk flags ile kanıtlanır.
- Final’e terfi için `build_final_submission_families.py`, `package_final_families.py`, `validate_final_release.py` yeniden çalışır.
- Eski family metadata korunur.
