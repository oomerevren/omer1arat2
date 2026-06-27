# Final Ensemble & Submission Family Recommendation

## 13.1 En güvenli gönderim

**Family A — Balanced / Safest** varsayılan adaydır.

- Status: `not_ready_no_oof_or_test_predictions`
- Risk label: `safest_balanced`
- Models: ``
- Threshold: `None`

Neden: OOF-first, class0/class1 dengesi, düşük threshold oynaklığı ve public LB'ye aşırı bağımlı olmama prensibiyle tasarlanmıştır. Bir tek final hakkı varsa varsayılan seçim Family A'dır.

## 13.2 Agresif ama riskli gönderim

**Family C — Semantic / Recall / Aggressive** yalnızca risk bayrakları düşükse ve OOF/public sinyali birlikte olumluysa denenmelidir.

- Status: `not_ready_no_oof_or_test_predictions`
- Risk label: `semantic_aggressive`
- Models: ``

Dense/CE/semantic sinyaller private'da lift gösterebilir; ancak false-positive ve public-optimism riski nedeniyle ana varsayılan yapı değildir.

## 13.3 Public sinyal gelirse tercih edilecek gönderim

Family C'ye ancak şu koşullarda dönülür:

- Public artışı belirgin ve tekrarlanabilir.
- OOF macro-F1 düşmüyor.
- class0 F1 düşmüyor.
- `PUBLIC_UP_OOF_DOWN` veya `PUBLIC_UP_CLASS0_DOWN` bayrağı yok.
- Dense/retrieval artefact riski fold-safe recheck ile temiz.

Aldatıcı sayılacak public artışları:

- threshold tweak kaynaklı küçük artış
- OOF düşerken public artış
- class0 F1 düşerken public artış
- segment collapse pahasına gelen public artış

## 13.4 Son hafta karar protokolü

1. Önce Family A gönderilir / referans alınır.
2. Family B class0-private savunma alternatifi olarak saklanır.
3. Family C yalnızca semantic lift ve düşük risk kanıtlanırsa denenir.
4. Sinyaller karışıksa Family A korunur.
5. Public LB son sinyaldir; OOF + class0 + risk flags daha üstündür.

## Family B — defensive rolü

- Status: `not_ready_no_oof_or_test_predictions`
- Risk label: `private_defensive`
- Models: ``

Family B yanlış pozitifleri azaltma, class0 F1'i koruma ve private irrelevant segment çöküşünü engelleme rolündedir.

## Blend summary

Tested blends: 0
Rejected blends: 1

## Freeze readiness

Family config dosyaları `configs/kaggle/final/` altında, metadata dosyaları `artifacts/final_submissions/` altında üretilmiştir. Official test predictions yoksa submission.csv bilinçli olarak materialize edilmez.
