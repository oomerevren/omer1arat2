# Final Repo Lockdown & Reproducibility Sprint

## Final config’ler nasıl ayrıldı?

Final family config’leri `configs/kaggle/final/` altında toplandı:

```text
shared_final_base.yaml
family_A_balanced.yaml
family_B_defensive.yaml
family_C_aggressive.yaml
```

Deneysel config alanı ayrı açıldı:

```text
configs/kaggle/experiments/
```

Final config’lerde `final_mode: true` ve `frozen: true` guard’ları bulunur. Config checksum’ları `reports/final/final_config_freeze_index.csv/json` içinde saklanır.

## Family’ler nasıl paketlendi?

Prompt 18 çıktıları standart final artefact alanına taşındı:

```text
artifacts/final/families/<family>/metadata.json
artifacts/final/families/<family>/selected_models.json
artifacts/final/families/<family>/threshold.json
artifacts/final/submissions/<family>_submission.csv
```

Official test predictions yoksa submission dosyası bilinçli olarak üretilmez; metadata yine freeze edilir.

## Artefact manifest ne içeriyor?

Ana manifest:

```text
reports/final/final_artifact_manifest.json
```

İçerik:

- family config path ve checksum
- final_mode/frozen durumu
- metadata path
- submission path
- models used
- source experiment ids
- blend weights
- threshold
- risk label
- validation report path
- registry bağlantıları

## Yarışma günü sıra

1. `python scripts_kaggle/analyze_leaderboard_correlation.py`
2. `python scripts_kaggle/build_final_submission_families.py --config configs/kaggle/war_mode.yaml`
3. `python scripts_kaggle/package_final_families.py`
4. `python scripts_kaggle/validate_final_release.py`
5. `docs/submission_day_checklist.md` üzerinden manuel onay
6. Kaggle upload
7. Public LB sonucu tracking table’a işlenir

## Freeze sonrası değişiklik nasıl yönetilecek?

- Final config doğrudan editlenmez.
- Yeni deneme `configs/kaggle/experiments/` altına alınır.
- Deneme OOF/class0/risk raporlarıyla kanıtlanır.
- Terfi gerekiyorsa final family builder ve package/freeze scriptleri tekrar çalışır.
- Manifest ve release validation raporu güncellenmeden yeni dosya final kabul edilmez.

## En riskli kalan alan

Official data/model artefact’ları workspace’te olmadığı için final metadata kilitli fakat real `submission.csv` henüz materialize değildir. Bu, release validator tarafından warning olarak işaretlenir.
