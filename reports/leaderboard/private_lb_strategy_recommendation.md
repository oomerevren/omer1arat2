# Private Leaderboard Strategy Recommendation

## Karar sinyali önceliği

1. OOF macro-F1
2. class 0 F1
3. splitter reliability
4. threshold fragility
5. seed stability
6. segment collapse risk
7. model family drift
8. public LB

## En güvenilir validation splitter

Öneri: `term_group`

Recommendation is statistical only if n_public>=3; otherwise guarded default.

## Threshold stratejisi

Prefer OOF global best or stable midpoint. Do not tune threshold directly to public LB; segment thresholds are analysis-only unless stable across seeds.

## Model family notu

Do not rank families from public LB unless n_public is sufficient; use OOF/class0/seed stability first.

## Safest balanced candidates

No private_safe candidate yet; official OOF/public history is insufficient.

## Balanced candidates

No balanced candidate yet.

## Public-optimistic / dikkat

No public-optimistic candidate detected yet.

## Son hafta protokolü

- Public LB artışı OOF düşüşüyle gelirse final adayından çıkar.
- Class0 F1 düşerken public artıyorsa private için alarm kabul edilir.
- Threshold tweak kaynaklı public artışları model iyileşmesi sayılmaz.
- Dense/retrieval-heavy adaylar fold-safe recheck olmadan ana submission yapılmaz.
- En az bir balanced/private-safe ve bir class0-heavy savunmacı submission family saklanır.
