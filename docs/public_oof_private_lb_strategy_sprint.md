# Public / OOF / Private LB Strategy Sprint

Bu sprint public leaderboard skorlarını bir seçim oracle'ı olarak değil, OOF ve segment davranışıyla birlikte okunması gereken bir dış sinyal olarak ele alır.

## Public vs OOF nasıl bağlandı?

Yeni leaderboard modülü şu kaynakları normalize eder ve birleştirir:

```text
reports/experiments/experiment_registry.csv
reports/submissions/submission_registry.csv
reports/ablation/ablation_master_table.csv
reports/leaderboard/public_lb_tracking_table.csv
```

Manuel public LB girişi:

```bash
python scripts_kaggle/analyze_leaderboard_correlation.py \
  --add-public-entry \
  --experiment-id <EXP_ID> \
  --file-path submissions/x.csv \
  --public-lb-score 0.81234 \
  --notes "Kaggle public LB run"
```

Ana tablo:

```text
reports/leaderboard/oof_public_correlation.csv
```

## Hangi splitter daha güvenilir çıktı?

Sistem splitter bazında Pearson/Spearman, sign agreement, public-minus-OOF delta ve high-risk count üretir:

```text
reports/leaderboard/splitter_reliability_report.json
reports/leaderboard/splitter_reliability_summary.md
```

Official public/OOF geçmişi azsa sistem güçlü istatistik iddiası yapmaz. Guarded default: `term_group`, çünkü query/term leakage riskine karşı daha savunmacıdır.

## Hangi model family daha riskli çıktı?

Family raporları:

```text
reports/leaderboard/model_family_drift_report.json
reports/leaderboard/model_family_drift_summary.md
reports/leaderboard/model_family_comparison.csv
```

Özellikle dikkat edilen family'ler:

- tabular
- sklearn_text
- transformer_ce
- dense_enhanced
- retrieval_heavy
- ensemble

Dense/retrieval-heavy ve ensemble adayları public-minus-OOF deltasında aşırı optimistic görünürse private risk flag alır.

## Public artışı ne zaman ciddiye alınmalı?

Public artışı ancak şu koşullarda ciddiye alınır:

1. OOF macro-F1 de artıyor.
2. class0 F1 düşmüyor.
3. threshold kırılgan değil.
4. seed/splitter stabil.
5. segment collapse yok.
6. Artış yalnızca threshold tweak kaynaklı değil.
7. Dense/retrieval artefact riski fold-safe recheck ile temiz.

## Final aileleri kurarken hangi sinyallere güvenilecek?

Karar önceliği:

1. OOF macro-F1
2. class 0 F1
3. splitter reliability
4. threshold fragility
5. seed stability
6. segment collapse risk
7. model family drift
8. public LB

Public LB en son sinyaldir; final aileleri Prompt 18'de bu karar sırasıyla kurulmalıdır.

## Önemli risk flag'ler

```text
PUBLIC_UP_OOF_DOWN
PUBLIC_UP_CLASS0_DOWN
THRESHOLD_FRAGILE
SPLITTER_INCONSISTENT
SEGMENT_COLLAPSE_RISK
ENSEMBLE_OVERFIT_RISK
DENSE_ARTIFACT_RISK
RETRIEVAL_FEATURE_DRIFT
PRIVATE_UNSAFE_CANDIDATE
```

Final hafta protokolü: `PUBLIC_UP_CLASS0_DOWN` veya `PUBLIC_UP_OOF_DOWN` taşıyan submission ana final adayı yapılmaz.
