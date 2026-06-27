# Final Ensemble & Submission Family Sprint

Bu sprint tek bir “en iyi model” seçmek yerine private leaderboard belirsizliğine karşı üç stratejik submission ailesi kurar.

## Final havuza hangi adaylar alınır?

Adaylar şu tablodan seçilir:

```text
reports/leaderboard/oof_public_correlation.csv
```

Seçim önceliği:

1. OOF macro-F1
2. class0 F1
3. threshold stability
4. seed stability
5. public/private risk flags
6. public LB en son sinyal

Çıktı:

```text
reports/final/final_candidate_pool.csv
```

Official OOF yoksa sistem skor uydurmaz; `not_ready_no_oof` placeholder adayları üretir.

## Hangi blend'ler denenir?

Öncelik sırası:

1. Weighted average
2. Rank average
3. Stacking yalnızca yeterli güçlü OOF geçmişi varsa ileride

Blend karşılaştırması:

```text
reports/final/final_blend_comparison.json
reports/final/final_blend_comparison.md
```

Yüksek korelasyonlu ve yeni bilgi katmayan blend'ler rejected listesine alınır.

## Neden 3 family seçildi?

Private LB belirsizliğinde tek submission mantığı risklidir. Bu yüzden üç rol ayrılır:

### Family A — Balanced / Safest

Varsayılan final adayı. OOF/class0/class1 dengesi ve düşük risk odaklıdır.

### Family B — Class0 / Precision / Defensive

Yanlış pozitifleri azaltmak ve irrelevant sınıfını korumak için tasarlanır. Public'de biraz daha düşük görünse bile private savunması sağlar.

### Family C — Semantic / Recall / Aggressive

Dense/CE/semantic sinyalleri daha yüksek kullanır. Public ve OOF birlikte olumluysa challenger olarak denenir; varsayılan değildir.

## Private LB açısından en güvenli olan hangisi?

Varsayılan cevap Family A'dır. Family B private class0 çöküş riskine karşı savunmacı alternatiftir. Family C ancak risk flags temizse değerlendirilir.

## Public sinyali gelirse nasıl hareket edilir?

- Public artıyor ama OOF/class0 düşüyorsa Family A korunur.
- Public artışı class0-safe ve OOF-consistent ise Family B/A güçlenir.
- Semantic family public'de belirgin artıyor, OOF düşmüyor ve risk flags düşükse Family C sınırlı riskle denenebilir.
- Threshold tweak kaynaklı küçük public artış model iyileşmesi sayılmaz.

## Tekrar üretilebilir artefact yapısı

```text
artifacts/final_submissions/family_A_balanced/metadata.json
artifacts/final_submissions/family_B_defensive/metadata.json
artifacts/final_submissions/family_C_aggressive/metadata.json
configs/kaggle/final/family_A_balanced.yaml
configs/kaggle/final/family_B_defensive.yaml
configs/kaggle/final/family_C_aggressive.yaml
```

Official test predictions ve `submission_pairs.csv` hazır olduğunda aynı pipeline valid `submission.csv` de üretir ve submission registry'ye işler.
