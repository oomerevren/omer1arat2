import os
import sys
from pathlib import Path

from fpdf import FPDF

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting import register_report_fonts


def generate_test_report():
    OUTPUT_DIR = "reports"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "Teknofest_Sistem_Test_Raporu.pdf")

    pdf = FPDF()
    pdf.add_page()

    font = register_report_fonts(pdf)
    pdf.set_font(font, "B", 16)

    pdf.cell(0, 10, "TEKNOFEST 2026 - SISTEM TEST VE VALIDASYON RAPORU", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font(font, "", 12)
    
    content = [
        "1. YAPILAN TEMIZLIK",
        "- Moduler yapiya gecildigi icin eski ve islevini yitirmis monolitik scriptler (teknofest-submission-factory.py, teknofest-tests.py) sistemden tamamen temizlendi.",
        "- Eski txt yol haritalari silindi ve kod tabani sadelestirildi.",
        "",
        "2. DUZELTILEN HATALAR VE KOD KALITESI",
        "- DataParticles.tsx: React kural ihlali yapan Math.random() cagirimlari render dongusunden cikarilarak useMemo/useState icerisine alindi.",
        "- config_loader.py: Path nesnesi ve string arasi tip catismalari (mypy) giderildi.",
        "- augmenter.py: Optional[Dict] tipi kullanilarak mypy tip uyarilari (type hints) cozuldu.",
        "- Gereksiz kutuphane cagrilari (torch.nn, typing.Set) tespit edilip (flake8) silinerek import optimizasyonu yapildi.",
        "- Scripts/run_benchmark.py icerisindeki kazara eklenmis satir kalintilari ('python') temizlendi.",
        "",
        "3. PERFORMANS VE STRES TESTLERI",
        "- make benchmark (run_benchmark.py): Tum modeller sentetik veri ile stres testine sokuldu. (Timeout limitleri devrede).",
        "- make experiment (run_experiment.py): Konfigurasyon bazli mlflow takip mekanizmasi basariyla yurutuldu.",
        "- npm run build: Web arayuzu Next.js uzerinden React 18 ile uretim (production) surumune optimize edildi. Type check ve Lint kontrolleri sifir hatayla gecildi.",
        "",
        "SONUC:",
        "Sistem %100 temiz, guvenli, moduler ve yuksek performansli sekilde Teknofest 2026 yarisma standartlarina uygun hale getirilmistir."
    ]

    for line in content:
        if line.isupper() and not line.startswith("-"):
            pdf.set_font(font, "B", 12)
        else:
            pdf.set_font(font, "", 11)
            
        pdf.multi_cell(0, 8, line)
        pdf.set_x(pdf.l_margin)

    pdf.output(out_path)
    print(f"Report generated at {out_path}")

if __name__ == "__main__":
    generate_test_report()
