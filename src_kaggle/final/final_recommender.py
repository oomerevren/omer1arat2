"""Final written recommendation for the three submission families."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def write_final_recommendation(families: pd.DataFrame, blend_report: dict, out_path: str | Path = "reports/final/final_recommendation.md") -> None:
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    def row(name):
        part = families[families["family_name"].eq(name)]
        return part.iloc[0].to_dict() if not part.empty else {}
    A, B, C = row("family_A_balanced"), row("family_B_defensive"), row("family_C_aggressive")
    lines = [
        "# Final Ensemble & Submission Family Recommendation", "",
        "## 13.1 En güvenli gönderim", "",
        "**Family A — Balanced / Safest** varsayılan adaydır.", "",
        f"- Status: `{A.get('status','missing')}`", f"- Risk label: `{A.get('public_private_risk_label','safest_balanced')}`", f"- Models: `{A.get('used_models','')}`", f"- Threshold: `{A.get('threshold','')}`", "",
        "Neden: OOF-first, class0/class1 dengesi, düşük threshold oynaklığı ve public LB'ye aşırı bağımlı olmama prensibiyle tasarlanmıştır. Bir tek final hakkı varsa varsayılan seçim Family A'dır.", "",
        "## 13.2 Agresif ama riskli gönderim", "",
        "**Family C — Semantic / Recall / Aggressive** yalnızca risk bayrakları düşükse ve OOF/public sinyali birlikte olumluysa denenmelidir.", "",
        f"- Status: `{C.get('status','missing')}`", f"- Risk label: `{C.get('public_private_risk_label','semantic_aggressive')}`", f"- Models: `{C.get('used_models','')}`", "",
        "Dense/CE/semantic sinyaller private'da lift gösterebilir; ancak false-positive ve public-optimism riski nedeniyle ana varsayılan yapı değildir.", "",
        "## 13.3 Public sinyal gelirse tercih edilecek gönderim", "",
        "Family C'ye ancak şu koşullarda dönülür:", "", "- Public artışı belirgin ve tekrarlanabilir.", "- OOF macro-F1 düşmüyor.", "- class0 F1 düşmüyor.", "- `PUBLIC_UP_OOF_DOWN` veya `PUBLIC_UP_CLASS0_DOWN` bayrağı yok.", "- Dense/retrieval artefact riski fold-safe recheck ile temiz.", "",
        "Aldatıcı sayılacak public artışları:", "", "- threshold tweak kaynaklı küçük artış", "- OOF düşerken public artış", "- class0 F1 düşerken public artış", "- segment collapse pahasına gelen public artış", "",
        "## 13.4 Son hafta karar protokolü", "", "1. Önce Family A gönderilir / referans alınır.", "2. Family B class0-private savunma alternatifi olarak saklanır.", "3. Family C yalnızca semantic lift ve düşük risk kanıtlanırsa denenir.", "4. Sinyaller karışıksa Family A korunur.", "5. Public LB son sinyaldir; OOF + class0 + risk flags daha üstündür.", "",
        "## Family B — defensive rolü", "", f"- Status: `{B.get('status','missing')}`", f"- Risk label: `{B.get('public_private_risk_label','private_defensive')}`", f"- Models: `{B.get('used_models','')}`", "", "Family B yanlış pozitifleri azaltma, class0 F1'i koruma ve private irrelevant segment çöküşünü engelleme rolündedir.", "",
        "## Blend summary", "", f"Tested blends: {len(blend_report.get('tested_blends', []))}", f"Rejected blends: {len(blend_report.get('rejected_blends', []))}", "",
        "## Freeze readiness", "", "Family config dosyaları `configs/kaggle/final/` altında, metadata dosyaları `artifacts/final_submissions/` altında üretilmiştir. Official test predictions yoksa submission.csv bilinçli olarak materialize edilmez.",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
