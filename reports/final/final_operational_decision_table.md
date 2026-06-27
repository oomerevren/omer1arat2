# Final Operational Decision Table

| Situation | Action | Family | Stop conditions |
| --- | --- | --- | --- |
| Default / signals mixed | Use balanced private-safe candidate | Family A | Any validator failure stops submission |
| Class0 F1 pressure, public gain suspicious | Prefer defensive candidate | Family B | If recall collapse/segment collapse is severe, revert to A |
| Semantic family has OOF + public lift and low risk flags | Consider aggressive challenger | Family C | PUBLIC_UP_OOF_DOWN, PUBLIC_UP_CLASS0_DOWN, THRESHOLD_FRAGILE, DENSE_ARTIFACT_RISK unresolved |
| Public LB tiny improvement only | Do not switch by public alone | Family A | public-only threshold tweak ignored |
| Submission validator warning/error | Do not upload until resolved | none | format/id/order/binary prediction errors are hard stop |

## Hard stop risk flags

- PUBLIC_UP_OOF_DOWN
- PUBLIC_UP_CLASS0_DOWN
- PRIVATE_UNSAFE_CANDIDATE
- missing submission validator report
- config final_mode is not true
- manifest/config checksum mismatch

## Default final answer

If the team cannot reach consensus, keep **Family A — Balanced / Safest**.
