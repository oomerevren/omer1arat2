"""
Teknofest 2026 Deep-Pipeline — Düzeltme Raporu (PDF)
17 Haziran 2026 eksiklik raporuna göre yapılan düzeltmeler.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting import register_report_fonts

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOW = datetime.now()
DATE_STR = NOW.strftime("%Y-%m-%d")
TIME_STR = NOW.strftime("%H-%M")
FILENAME = f"Teknofest_Duzeltme_Raporu_{DATE_STR}_{TIME_STR}.pdf"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, FILENAME)


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=22)
        register_report_fonts(self)

    def header(self):
        self.set_font("ArialUni", "B", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, "TEKNOFEST 2026 | Deep-Pipeline | Duzeltme Raporu", align="L")
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.set_font("ArialUni", "I", 8)
        self.cell(0, 8, f"Sayfa {self.page_no()}/{{nb}} | {DATE_STR}", align="C")

    def section_title(self, text):
        self.set_font("ArialUni", "B", 14)
        self.set_text_color(20, 20, 20)
        self.ln(3)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body(self, text):
        self.set_font("ArialUni", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("ArialUni", "", 10)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, "- " + text)
        self.ln(0.3)


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("ArialUni", "B", 20)
    pdf.cell(0, 12, "Duzeltme Raporu", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("ArialUni", "", 11)
    pdf.cell(0, 7, f"Tarih: {DATE_STR} {NOW.strftime('%H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Referans: eksikler/eksiklik-raporu_17-06-2026.pdf", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.section_title("1. Duzeltilen Eksikler")
    fixed = [
        "Kaggle modu label bug: normalize_labels + sample_generator binary donusum",
        "Distillation trainer.py entegrasyonu (run_distillation)",
        "CrossValidator egitim oncesi CV metrikleri",
        "Quantization trainer save_model entegrasyonu",
        "Ensemble stacking (LogisticRegression meta) + Optuna agirlik aramasi",
        "Pseudo-labeling trainer entegrasyonu",
        "XAI: transformer attention aciklamasi (get_attention_explanation)",
        "Negative mining: embedding_nearby, same_brand, config stratejileri",
        "Category-specific threshold aramasi",
        "API /metrics: gercek memory (psutil), metrics.json F1",
        "Benchmark: sentetik mod varsayilan degil (--synthetic ile acilir)",
        "Kaggle submission: resolve_model_path, metrics.json threshold",
        "DetailedModules kanitsiz %98 accuracy iddiasi kaldirildi",
        "E-commerce SymSpell sozlugu (data/ecommerce_dict.txt)",
        "Data augmentation trainer entegrasyonu",
        "Dockerfile offline stratejisi (LOCAL_MODEL_PATH)",
        "KYS basvuru: docs/Takim_Tanitim.md",
        "Gorev dagilimi: docs/Takim_Gorev_Dagilimi.md",
        "Kaggle notebook rehberi: notebooks/KAGGLE_BASELINE.md",
        "XAI demo script: scripts/run_xai_demo.py",
        "Final rapor taslag: docs/Final_Rapor_Taslagi.md",
        "CPU plani: docs/Kaggle_CPU_Plani.md",
        "26 Haziran veri hazirligi: scripts/prepare_kaggle_data.py",
        "E2E API testleri: tests/test_api.py (15 test geciyor)",
        "Label testleri: tests/test_labels.py",
        "cross_encoder save: safetensors fallback (Windows erisim hatasi)",
        "README guncellendi",
    ]
    for item in fixed:
        pdf.bullet(item)

    pdf.add_page()
    pdf.section_title("2. Duzeltilemeyen / Ertelenen Eksikler")
    deferred = [
        "Gercek Kaggle egitim verisi — 26 Haziran 2026",
        "Gercek test verisi ve submission skoru — 26 Haziran 2026",
        "Private leaderboard performansi — veri sonrasi",
        "Final set 3-sinif performans kaniti — final veri seti sonrasi",
        "Hiperparametre optimizasyonu (tam Optuna HPO) — veri sonrasi",
        "Gercek benchmark Macro-F1 karsilastirmasi — veri sonrasi",
        "Tam feature ablation — veri sonrasi",
        "KYS yukleme islemi — takim tarafindan manuel",
    ]
    for item in deferred:
        pdf.bullet(item)

    pdf.section_title("3. Neden Duzeltilemedi")
    pdf.body(
        "Organizator egitim/test verisi 26 Haziran 2026'da iletilecek. Macro-F1, "
        "leaderboard siralamasi ve nihai performans degerlendirmesi gercek veri "
        "olmadan anlamli degildir. Bu maddeler icin altyapi (pipeline, script, config) "
        "hazirlanmistir; veri gelince calistirilmaya hazirdir."
    )
    pdf.body(
        "KYS basvuru dosyasi icerik olarak docs/ altinda olusturuldu; www.t3kys.com "
        "yuklemesi takim uyelerinin hesabi ile yapilmalidir (dis sistem erisimi)."
    )

    pdf.section_title("4. 26 Haziran Sonrasi Yapilacaklar")
    post = [
        "prepare_kaggle_data.py ile veriyi data/ klasorune yerlestir",
        "EDA + normalize_dataframe dogrulama",
        "run_experiment.py --config configs/model/kaggle.yaml",
        "Gunluk kaggle_submission.py ile LB takibi",
        "Benchmark gercek sonuclarla guncelle (make benchmark)",
        "Hiperparametre / ensemble Optuna iterasyonu",
        "Finalist olursa: 3-sinif model, distillation+quantization, final rapor",
    ]
    for item in post:
        pdf.bullet(item)

    pdf.add_page()
    pdf.section_title("5. Guncellenen Dosyalar")
    updated = [
        "src/training/trainer.py", "src/training/distillation.py", "src/training/pseudo_labeler.py",
        "src/models/cross_encoder.py", "src/models/ensemble.py",
        "src/data/dataset.py", "src/data/labels.py", "src/data/negative_miner.py",
        "src/data/sample_generator.py", "src/data/preprocessor.py",
        "src/evaluation/threshold_search.py", "src/xai/explainer.py",
        "src/deployment/api.py", "configs/base_config.yaml", "configs/model/kaggle.yaml",
        "scripts/kaggle_submission.py", "scripts/run_benchmark.py",
        "deep-pipeline-web/.../DetailedModules.tsx", "Dockerfile", "Makefile", "README.md",
    ]
    for f in updated:
        pdf.bullet(f)

    pdf.section_title("6. Yeni Eklenen Dosyalar")
    new_files = [
        "src/data/labels.py", "data/ecommerce_dict.txt",
        "scripts/prepare_kaggle_data.py", "scripts/run_xai_demo.py",
        "scripts/generate_duzeltme_raporu.py",
        "tests/test_api.py", "tests/test_labels.py",
        "docs/Takim_Tanitim.md", "docs/Takim_Gorev_Dagilimi.md",
        "docs/Final_Rapor_Taslagi.md", "docs/Kaggle_CPU_Plani.md",
        "notebooks/KAGGLE_BASELINE.md",
    ]
    for f in new_files:
        pdf.bullet(f)

    pdf.section_title("7. Silinen Dosyalar")
    pdf.body("Silinen dosya yok. Sentetik benchmark raporu gecersiz sayilmali; make benchmark ile yeniden uretilecek.")

    pdf.section_title("8. Yeni Mimari Ozeti")
    pdf.body(
        "Veri katmani (normalize_labels, augmentation, negative mining) -> Egitim "
        "(CV raporu, distillation veya CE/ensemble, pseudo-label, threshold) -> "
        "Kayit (model + metrics.json + quantization) -> Servis (FastAPI predict/explain) -> "
        "XAI (feature + attention) -> Sunum (Streamlit, Next.js). "
        "26 Haziran sonrasi prepare_kaggle_data.py tek giris noktasi."
    )

    pdf.add_page()
    pdf.section_title("9. Guncel Risk Analizi")
    pdf.bullet("DUSUK: Pipeline butunlugu — 15 test geciyor")
    pdf.bullet("ORTA: Gercek veri gelene kadar LB performansi bilinmiyor")
    pdf.bullet("ORTA: Distillation CPU'da yavas — Kaggle'da kapali")
    pdf.bullet("DUSUK: Tutarlilik riski — sentetik metrikler kaldirildi/etiketlendi")

    pdf.section_title("10. Sartname Uyumluluk Yuzdesi")
    pdf.body("Altyapi ve dokumantasyon: ~78% | Performans kaniti: ~15% | Agirlikli toplam: ~62%")
    pdf.body("(Performans maddeleri veri gelince ~90%+ hedefleniyor)")

    pdf.section_title("11. Teknik Puan Tahmini (Final)")
    pdf.body("Bugun (veri oncesi): 55-65/100 | Veri + iyi Kaggle sonrasi: 75-85/100")

    pdf.section_title("12. Yenilikcilik Puani Tahmini")
    pdf.body("Turkce NLP pipeline + attention XAI + uctan uca repo: 7/10")

    pdf.section_title("13. Juri Acisindan Guclu Yonler")
    pdf.bullet("Calisan uctan uca sistem (train -> API -> XAI -> UI)")
    pdf.bullet("Acik kaynak, offline hazirlik, reproducible scriptler")
    pdf.bullet("Negatif mining + kategori esigi + ensemble derinligi")

    pdf.section_title("14. Juri Acisindan Zayif Yonler")
    pdf.bullet("Henuz gercek Kaggle skoru yok")
    pdf.bullet("DistilBERTurk baseline henuz organizator verisiyle egitilmedi")
    pdf.bullet("SHAP yok (attention + feature var)")

    pdf.section_title("15. 1. Lik Ihtimali (Objektif)")
    pdf.body(
        "Bugun: 2.5/10 (performans kaniti yok). Veri sonrasi iyi optimizasyon ile: "
        "6-7/10. 1. lik icin Kaggle ust sira + final set + hiz + XAI + sunum birlikte gerekir."
    )

    pdf.section_title("16. Teknik Gerekce — Kismen Katilmadigimiz Maddeler")
    pdf.body(
        "Eksiklik raporundaki 'SHAP zorunlu' yorumu: Attention weights + feature "
        "aciklamasi juri demosu icin yeterli baslangic; SHAP ek yuk getirir, veri "
        "sonrasi eklenebilir. 'Stacking 0.874 F1' maddesi: Sentetik degerler "
        "silindi/kullanilmiyor; gercek stacking implementasyonu eklendi."
    )

    pdf.output(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_report()
    print(f"Rapor: {path}")
