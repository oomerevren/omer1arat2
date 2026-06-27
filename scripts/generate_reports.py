"""
Teknofest 2026 - Deep Pipeline
Resmi PDF Rapor Uretici (Unicode / Turkce Destekli)

Bu script, Teknofest yarismasi icin iki adet resmi PDF raporu uretir:
1. Sistem Tasarim Raporu (PDR)
2. Gelistirme Surec Raporu
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting import register_report_fonts

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TeknofestPDF(FPDF):
    """Teknofest stilinde PDF uretici (Unicode destekli)."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)
        register_report_fonts(self, courier=True)

    def header(self):
        self.set_font("ArialUni", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "TEKNOFEST 2026 | Deep-Pipeline | E-Ticaret Hackathonu", align="L")
        self.ln(3)
        self.set_draw_color(255, 215, 0)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("ArialUni", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("ArialUni", "B", 16)
        self.set_text_color(10, 10, 10)
        self.ln(5)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(255, 215, 0)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.set_font("ArialUni", "B", 13)
        self.set_text_color(40, 40, 40)
        self.ln(3)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("ArialUni", "", 11)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6.5, text)
        self.ln(3)

    def bullet_point(self, text):
        self.set_font("ArialUni", "", 11)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6.5, "  -  " + text)

    def code_block(self, text):
        self.set_fill_color(240, 240, 240)
        self.set_font("CourierUni", "", 9)
        self.set_text_color(50, 50, 50)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, text, fill=True)
        self.ln(3)

    def cover_page(self, title, subtitle):
        self.add_page()
        self.ln(60)
        self.set_font("ArialUni", "B", 28)
        self.set_text_color(10, 10, 10)
        self.multi_cell(0, 14, title, align="C")
        self.ln(8)
        self.set_font("ArialUni", "", 16)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 10, subtitle, align="C")
        self.ln(15)
        self.set_draw_color(255, 215, 0)
        self.set_line_width(1.5)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)
        self.set_font("ArialUni", "", 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "TEKNOFEST 2026 \u2014 E-Ticaret Hackathonu", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, f"Tarih: {datetime.now().strftime('%d.%m.%Y')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "Tak\u0131m: Deep-Pipeline", align="C", new_x="LMARGIN", new_y="NEXT")


def generate_system_design_report():
    """Rapor 1: Sistem Tasar\u0131m Raporu (PDR)"""
    pdf = TeknofestPDF()
    pdf.alias_nb_pages()

    # \u2500\u2500 KAPAK \u2500\u2500
    pdf.cover_page(
        "DEEP-PIPELINE",
        "E-Ticaret Semantik Arama ve\nVeri M\u00fchendisli\u011fi Ekosistemi\nSistem Tasar\u0131m Raporu (PDR)"
    )

    # \u2500\u2500 1. PROJE \u00d6ZET\u0130 \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("1. Proje \u00d6zeti")
    pdf.body_text(
        "Deep-Pipeline, e-ticaret platformlar\u0131ndaki \u00fcr\u00fcn arama s\u00fcrecini "
        "yeniden tan\u0131mlayan, u\u00e7tan uca bir semantik arama ve veri m\u00fchendisli\u011fi "
        "ekosistemidir. Proje, kullan\u0131c\u0131lar\u0131n do\u011fal dilde girdikleri arama "
        "sorgular\u0131n\u0131 anlamland\u0131rarak, en uygun \u00fcr\u00fcnlerle e\u015fle\u015ftirir. "
        "Geleneksel anahtar kelime tabanl\u0131 (keyword-based) arama motorlar\u0131n\u0131n "
        "aksine, Deep-Pipeline do\u011fal dil i\u015fleme (NLP) ve derin \u00f6\u011frenme (Deep "
        "Learning) tekniklerini kullanarak semantik anlam \u00e7\u0131karmas\u0131 yapar."
    )
    pdf.body_text(
        "Sistem, Teacher-Student Distillation mimarisi sayesinde b\u00fcy\u00fck dil "
        "modellerinin do\u011frulu\u011funun (BGE-M3) k\u00fc\u00e7\u00fck ve h\u0131zl\u0131 modellere "
        "(DistilBERTurk) aktar\u0131lmas\u0131n\u0131 sa\u011flar. Sonu\u00e7 olarak, milisaniye "
        "seviyesinde yan\u0131t s\u00fcresiyle y\u00fcksek do\u011fruluklu sonu\u00e7lar sunar."
    )

    # \u2500\u2500 2. PROBLEM DURUMU \u2500\u2500
    pdf.chapter_title("2. Problem Durumu ve \u00c7\u00f6z\u00fcm")
    pdf.section_title("2.1 Problem")
    pdf.body_text(
        "E-ticaret platformlar\u0131nda g\u00fcnl\u00fck milyonlarca arama sorgusu "
        "ger\u00e7ekle\u015ftirilmektedir. Bu sorgular\u0131n b\u00fcy\u00fck \u00e7o\u011funlu\u011fu yaz\u0131m hatalar\u0131, "
        "eksik bilgiler veya belirsiz ifadeler i\u00e7erir. \u00d6rne\u011fin, bir kullan\u0131c\u0131 "
        "'siyah deri erkek c\u00fczdan' ararken, 'Derimod Hakiki Deri C\u00fczdanl\u0131k' "
        "\u00fcr\u00fcn\u00fcn\u00fc bulamayabilir. Bu durum, hem kullan\u0131c\u0131 deneyimini olumsuz "
        "etkiler hem de platformlar\u0131n gelir kayb\u0131 ya\u015famas\u0131na neden olur."
    )
    pdf.body_text(
        "Mevcut sistemlerin kar\u015f\u0131la\u015ft\u0131\u011f\u0131 temel sorunlar \u015funlard\u0131r:"
    )
    pdf.bullet_point("Yaz\u0131m hatalar\u0131na kar\u015f\u0131 duyarl\u0131l\u0131k (\u00d6rnek: 'aypon' vs 'Apple')")
    pdf.bullet_point("Marka-\u00fcr\u00fcn uyumsuzlu\u011fu (brand mismatch)")
    pdf.bullet_point("Aksesuar ve ana \u00fcr\u00fcn kar\u0131\u015f\u0131kl\u0131\u011f\u0131 (accessory confusion)")
    pdf.bullet_point("Kategoriler aras\u0131 yanl\u0131\u015f e\u015fle\u015fmeler (cross-category errors)")
    pdf.bullet_point("D\u00fc\u015f\u00fck hat\u0131rlanabilirlik (recall) oran\u0131 nadir kategorilerde")

    pdf.section_title("2.2 \u00c7\u00f6z\u00fcm: Deep-Pipeline Ekosistemi")
    pdf.body_text(
        "Deep-Pipeline, bu problemleri be\u015f ana mod\u00fclle \u00e7\u00f6zer: "
        "(1) Zemberek/Zeyrek tabanl\u0131 T\u00fcrk\u00e7e metin \u00f6n i\u015fleme, "
        "(2) RapidFuzz ile fuzzy marka e\u015fle\u015ftirme, "
        "(3) Hard Negative Mining ile zorlayıcı e\u011fitim \u00f6rnekleri, "
        "(4) Teacher-Student Distillation (Margin-MSE) ile bilgi aktarımı, "
        "(5) Kategori bazlı dinamik e\u015fik de\u011fer (threshold) optimizasyonu."
    )

    # \u2500\u2500 3. Y\u00d6NTEM \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("3. Y\u00f6ntem (Methodology)")
    pdf.section_title("3.1 Veri Seti ve \u00d6n \u0130\u015fleme")
    pdf.body_text(
        "Yar\u0131\u015fma kapsam\u0131nda sa\u011flanan e-ticaret veri seti, arama sorgular\u0131 ve "
        "\u00fcr\u00fcn bilgilerini (ba\u015fl\u0131k, marka, kategori) i\u00e7erir. Her bir sorgu-\u00fcr\u00fcn "
        "\u00e7ifti, 0 (Alakas\u0131z), 1 (K\u0131smen Alakal\u0131) veya 2 (\u00c7ok Alakal\u0131) olarak "
        "etiketlenmi\u015ftir. Veri \u00f6n i\u015fleme pipeline'\u0131 \u015funlar\u0131 kapsar:"
    )
    pdf.bullet_point("T\u00fcrk\u00e7e \u00f6zel k\u00fc\u00e7\u00fck harf d\u00f6n\u00fc\u015f\u00fcm\u00fc (\u0130/i ve I/\u0131 \u00e7ak\u0131\u015fmas\u0131)")
    pdf.bullet_point("Noktalama i\u015faretlerinin \u00e7\u0131kar\u0131lmas\u0131")
    pdf.bullet_point("SymSpell ile yaz\u0131m hatas\u0131 d\u00fczeltme (typo correction)")
    pdf.bullet_point("Zeyrek (Zemberek) ile morfolojik analiz ve lemmatizasyon")

    pdf.section_title("3.2 Model Mimarisi")
    pdf.body_text(
        "Sistemin \u00e7ekirde\u011finde Cross-Encoder mimarisi bulunmaktad\u0131r. "
        "Cross-Encoder, sorgu ve \u00fcr\u00fcn metnini tek bir girdi olarak al\u0131r "
        "([CLS] sorgu [SEP] \u00fcr\u00fcn [SEP]) ve do\u011frudan relevans skorunu \u00fcretir. "
        "Bu yakla\u015f\u0131m, Bi-Encoder modellerine k\u0131yasla \u00e7ok daha y\u00fcksek do\u011fruluk "
        "sa\u011flamaktad\u0131r."
    )
    pdf.body_text(
        "Teacher Model: BGE-M3 (BAAI) \u2014 \u00c7ok dilli, y\u00fcksek kapasiteli model.\n"
        "Student Model: DistilBERTurk \u2014 H\u0131zl\u0131, T\u00fcrk\u00e7e odakl\u0131 hafif model.\n"
        "Distillation Loss: Margin-MSE (Pairwise ranking sinyali).\n"
        "Ensemble: CatBoost + LightGBM + Cross-Encoder soft voting."
    )

    pdf.section_title("3.3 Hard Negative Mining")
    pdf.body_text(
        "Modelin ger\u00e7ek d\u00fcnyada kar\u015f\u0131la\u015faca\u011f\u0131 en zor senaryolara haz\u0131rlanmas\u0131 "
        "i\u00e7in \u00fc\u00e7 tip hard negative \u00f6rne\u011fi kullan\u0131lmaktad\u0131r: "
        "(1) Ayn\u0131 kategoriye ait fakat farkl\u0131 markadan \u00fcr\u00fcnler (In-Category), "
        "(2) BM25 ile y\u00fcksek kelime \u00f6rt\u00fc\u015fme oran\u0131na sahip ancak alakas\u0131z \u00fcr\u00fcnler, "
        "(3) Rassal negatif \u00f6rnekler (baseline)."
    )

    pdf.section_title("3.4 Teacher-Student Distillation (Margin-MSE)")
    pdf.body_text(
        "Bilgi aktar\u0131m\u0131 (Knowledge Distillation) s\u00fcrecinde, b\u00fcy\u00fck Teacher model "
        "(BGE-M3) taraf\u0131ndan \u00fcretilen soft-label'\u0131 \u00f6\u011frenci model (\u00f6\u011frenci) "
        "taklit etmeye \u00e7al\u0131\u015f\u0131r. Ancak klasik MSE loss yerine Margin-MSE "
        "kullan\u0131lmaktad\u0131r. Margin-MSE, iki \u00fcr\u00fcn aras\u0131ndaki skor fark\u0131n\u0131n "
        "(margin) Teacher ve Student modellerinde tutarl\u0131 olmas\u0131n\u0131 hedefler."
    )
    pdf.body_text(
        "Final Loss = \u03b1 \u00d7 CrossEntropyLoss + (1 \u2212 \u03b1) \u00d7 MarginMSELoss\n"
        "\u03b1 = 0.7 (base_config.yaml'\u0131 de\u011fi\u015ftirerek ayarlanabilir)"
    )

    # \u2500\u2500 4. S\u0130STEM TASARIMI \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("4. Sistem Tasar\u0131m\u0131 ve Mimari")
    pdf.body_text(
        "Proje, mod\u00fcler ve konfig\u00fcrasyon odakl\u0131 (config-driven) bir mimari "
        "\u00fczerine in\u015fa edilmi\u015ftir. T\u00fcm deney parametreleri YAML dosyalar\u0131ndan "
        "okunur, bu sayede kod de\u011fi\u015fikli\u011fi yapmadan y\u00fczlerce farkl\u0131 deney "
        "konfig\u00fcrasyonu test edilebilir."
    )

    pdf.section_title("4.1 Dosya Yap\u0131s\u0131")
    pdf.code_block(
        "deep-pipeline/\n"
        "  configs/\n"
        "    base_config.yaml       # Ana konfigurasyon\n"
        "    model/distilberturk.yaml\n"
        "  src/\n"
        "    data/preprocessor.py   # Metin on isleme\n"
        "    features/brand_features.py  # Fuzzy eslestirme\n"
        "    models/cross_encoder.py     # PyTorch modeli\n"
        "    training/distillation.py    # Teacher-Student\n"
        "    experiment/tracker.py       # MLflow entegrasyonu\n"
        "    experiment/config_loader.py # YAML cozumleyici\n"
        "  scripts/\n"
        "    run_experiment.py      # Deney baslatici\n"
        "    run_benchmark.py       # Benchmark testleri\n"
        "    run_dashboard.py       # Streamlit XAI\n"
        "  deep-pipeline-web/       # Next.js 3D sunum sitesi\n"
        "  Dockerfile               # Offline dagitim\n"
        "  requirements.txt"
    )

    pdf.section_title("4.2 Da\u011f\u0131t\u0131m ve Offline \u00c7al\u0131\u015fma")
    pdf.body_text(
        "Yar\u0131\u015fman\u0131n final a\u015famas\u0131nda modellerin internet eri\u015fimi olmayan bir "
        "cihazda \u00e7al\u0131\u015ft\u0131r\u0131lmas\u0131 gerekmektedir. Bu gereksinimi kar\u015f\u0131lamak i\u00e7in "
        "Docker tabanl\u0131 bir da\u011f\u0131t\u0131m stratejisi benimsenmi\u015ftir. Dockerfile "
        "i\u00e7erisinde TRANSFORMERS_OFFLINE=1 ve HF_DATASETS_OFFLINE=1 ortam "
        "de\u011fi\u015fkenleri tan\u0131mlanm\u0131\u015f olup, model a\u011f\u0131rl\u0131klar\u0131 lokal olarak "
        "/app/local_model dizininden y\u00fcklenmektedir."
    )

    # \u2500\u2500 5. YENILIK\u00c7\u0130 Y\u00d6N \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("5. Yenilik\u00e7i (\u0130novatif) Y\u00f6n")
    pdf.body_text(
        "Deep-Pipeline, a\u015fa\u011f\u0131daki y\u00f6nleriyle literat\u00fcr\u00fcn ve mevcut "
        "\u00e7\u00f6z\u00fcmlerin \u00f6tesine ge\u00e7mektedir:"
    )
    pdf.bullet_point(
        "Margin-MSE Distillation: Klasik MSE yerine pairwise ranking "
        "sinyali ta\u015f\u0131yan Margin-MSE loss fonksiyonu ile Teacher-Student "
        "aktar\u0131m\u0131. Bu, \u00f6\u011frenci modelin sadece skor tahmini de\u011fil, \u00fcr\u00fcnler "
        "aras\u0131 s\u0131ralama bilgisini de \u00f6\u011frenmesini sa\u011flar."
    )
    pdf.bullet_point(
        "Konfig\u00fcrasyon Odakl\u0131 MLOps: Tek bir YAML de\u011fi\u015fikli\u011fi ile model "
        "tipi, \u00f6\u011frenme oran\u0131, distillation katsay\u0131s\u0131, threshold de\u011ferleri "
        "ve veri seti yollar\u0131n\u0131 de\u011fi\u015ftirme imkan\u0131. Bu, yar\u0131\u015fmada g\u00fcnl\u00fck "
        "50+ deney yapmaya olanak tan\u0131r."
    )
    pdf.bullet_point(
        "A\u00e7\u0131klanabilir Yapay Zeka (XAI) Dashboard: Streamlit tabanl\u0131 "
        "interaktif panel ile modelin her bir karar\u0131n\u0131n arkas\u0131ndaki "
        "\u00f6zelliklerin (fuzzy skor, semantik benzerlik, kategori uyumu) "
        "\u015feffaf \u015fekilde g\u00f6sterilmesi."
    )
    pdf.bullet_point(
        "Brutalist 3D Web Deneyimi: Three.js ve Framer Motion ile "
        "tasarlanan, projenin teknik derinli\u011fini g\u00f6rsel olarak ileten, "
        "Teknik/Basit mod ge\u00e7i\u015fi ile farkl\u0131 kitlelere hitap eden sunum."
    )

    # \u2500\u2500 6. TEST VE PERFORMANS \u2500\u2500
    pdf.chapter_title("6. Test ve Performans Metrikleri")
    pdf.body_text(
        "Sistem performans\u0131 a\u015fa\u011f\u0131daki metrikler \u00fczerinden \u00f6l\u00e7\u00fclecektir. "
        "Yar\u0131\u015fma veri seti hen\u00fcz a\u00e7\u0131klanmad\u0131\u011f\u0131 i\u00e7in bu raporda kan\u0131ts\u0131z skor "
        "veya benchmark iddias\u0131 sunulmamaktad\u0131r. Nihai de\u011ferler organizat\u00f6r verisi "
        "ve final test ortam\u0131 sonras\u0131 doldurulacakt\u0131r:"
    )
    pdf.body_text(
        "Macro F1-Score:        Kaggle public/private leaderboard sonras\u0131 doldurulacak\n"
        "Inference Time:        Final servis testi sonras\u0131 doldurulacak\n"
        "VRAM/RAM Kullan\u0131m\u0131:  Final donan\u0131m testi sonras\u0131 doldurulacak\n"
        "QPS:                   Stress test sonras\u0131 doldurulacak"
    )
    pdf.body_text(
        "Do\u011frulama stratejisi olarak 5-fold StratifiedKFold kullan\u0131lacakt\u0131r. "
        "Her fold i\u00e7in Precision, Recall, F1-Score ve Confusion Matrix "
        "raporlanacakt\u0131r. Nihai skor, 5 seed ile Multi-Seed Ensemble "
        "ortalamas\u0131 olarak hesaplanacakt\u0131r."
    )

    # \u2500\u2500 7. UYGULANAB\u0130L\u0130RL\u0130K \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("7. Uygulanabilirlik ve Yayg\u0131n Etki")
    pdf.body_text(
        "Deep-Pipeline'\u0131n \u00fcretti\u011fi semantik arama modeli, T\u00fcrkiye'deki t\u00fcm "
        "e-ticaret platformlar\u0131na (Trendyol, Hepsiburada, n11 vb.) entegre "
        "edilebilir yap\u0131dad\u0131r. FastAPI \u00fczerinden sunulan REST API sayesinde, "
        "mevcut arama altyap\u0131lar\u0131na minimum de\u011fi\u015fiklikle eklenebilir."
    )
    pdf.body_text(
        "Projenin sosyal etkisi:\n"
        "\u2022 Kullan\u0131c\u0131 deneyiminin iyile\u015fmesi (daha do\u011fru sonu\u00e7lar)\n"
        "\u2022 E-ticaret platformlar\u0131n\u0131n gelir art\u0131\u015f\u0131 (azalan bounce rate)\n"
        "\u2022 T\u00fcrk\u00e7e NLP alan\u0131nda a\u00e7\u0131k kaynak katk\u0131\n"
        "\u2022 K\u00fc\u00e7\u00fck/orta \u00f6l\u00e7ekli e-ticaret sitelerine eri\u015filebilir AI \u00e7\u00f6z\u00fcm\u00fc"
    )

    # \u2500\u2500 8. PROJE TAKV\u0130M\u0130 \u2500\u2500
    pdf.chapter_title("8. Proje Takvimi")
    pdf.body_text(
        "Faz 1 (Haziran 2026): Altyap\u0131 kurulumu, MLOps pipeline, config sistemi.\n"
        "Faz 2 (26 Haziran \u2014 15 Temmuz): Veri analizi, baseline model e\u011fitimi.\n"
        "Faz 3 (15 Temmuz \u2014 15 A\u011fustos): Hard Negative Mining, Distillation.\n"
        "Faz 4 (15 A\u011fustos \u2014 1 Eyl\u00fcl): Ensemble, threshold optimizasyonu.\n"
        "Faz 5 (1 \u2014 15 Eyl\u00fcl): Final submitleri, rapor ve sunum haz\u0131rl\u0131k."
    )

    # \u2500\u2500 9. R\u0130SKLER \u2500\u2500
    pdf.chapter_title("9. Riskler ve Acil Durum Plan\u0131")
    pdf.body_text("Potansiyel riskler ve B planlar\u0131:")
    pdf.bullet_point(
        "Risk: Veri setinin dengesiz olmas\u0131 (class imbalance). "
        "Plan B: Focal Loss ve class weighting uygulanacak."
    )
    pdf.bullet_point(
        "Risk: Modelin overfitting yapmas\u0131. "
        "Plan B: Dropout art\u0131r\u0131m\u0131, data augmentation ve early stopping."
    )
    pdf.bullet_point(
        "Risk: Final cihaz\u0131nda GPU bulunmamas\u0131. "
        "Plan B: ONNX Runtime ile CPU-optimized inference."
    )

    # \u2500\u2500 10. REFERANSLAR \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("10. Referanslar")
    pdf.body_text(
        "[1] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence "
        "Embeddings using Siamese BERT-Networks. EMNLP.\n\n"
        "[2] Hofstatter, S. et al. (2021). Efficiently Teaching an Effective "
        "Dense Retriever with Balanced Topic Aware Sampling. SIGIR.\n\n"
        "[3] Khattab, O. & Zaharia, M. (2020). ColBERT: Efficient and "
        "Effective Passage Search via Contextualized Late Interaction. SIGIR.\n\n"
        "[4] Chen, J. et al. (2024). BGE M3-Embedding: Multi-Lingual, "
        "Multi-Functionality, Multi-Granularity. arXiv:2402.03216.\n\n"
        "[5] Schweter, S. (2020). BERTurk \u2014 BERT models for Turkish. "
        "Zenodo. doi:10.5281/zenodo.3770924.\n\n"
        "[6] Akbik, A. et al. (2019). FLAIR: An Easy-to-Use Framework for "
        "State-of-the-Art NLP. NAACL."
    )

    # \u2500\u2500 KAYDET \u2500\u2500
    out_path = os.path.join(OUTPUT_DIR, "Teknofest_PDR_Sistem_Tasarimi.pdf")
    pdf.output(out_path)
    print(f"  [OK] Sistem Tasar\u0131m Raporu olu\u015fturuldu: {out_path}")
    return out_path


def generate_development_report():
    """Rapor 2: Geli\u015ftirme S\u00fcrec Raporu"""
    pdf = TeknofestPDF()
    pdf.alias_nb_pages()

    # \u2500\u2500 KAPAK \u2500\u2500
    pdf.cover_page(
        "DEEP-PIPELINE",
        "Geli\u015ftirme ve S\u00fcrec\u0327 Raporu\nYaz\u0131l\u0131m M\u00fchendisli\u011fi Dok\u00fcmantasyonu"
    )

    # \u2500\u2500 1. G\u0130R\u0130\u015e \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("1. Giri\u015f ve Ama\u00e7")
    pdf.body_text(
        "Bu rapor, Deep-Pipeline projesinin yaz\u0131l\u0131m geli\u015ftirme s\u00fcrecini, "
        "al\u0131nan m\u00fchendislik kararlar\u0131n\u0131 ve uygulanan refactoring i\u015flemlerini "
        "detayl\u0131 \u015fekilde dok\u00fcmante etmektedir. Raporun amac\u0131, projenin "
        "ba\u015flang\u0131\u00e7 noktas\u0131ndan yar\u0131\u015fmaya haz\u0131r hale geli\u015fine kadarki t\u00fcm "
        "ad\u0131mlar\u0131 \u015feffaf bir \u015fekilde sunmakt\u0131r."
    )

    # \u2500\u2500 2. BA\u015eLANGI\u00c7 DURUMU \u2500\u2500
    pdf.chapter_title("2. Ba\u015flang\u0131\u00e7 Durumu (Refactoring \u00d6ncesi)")
    pdf.body_text(
        "Proje ba\u015flang\u0131c\u0131nda, sistem iki b\u00fcy\u00fck monolitik Python dosyas\u0131ndan "
        "olu\u015fuyordu: teknofest-benchmark-runner.py (1002 sat\u0131r) ve "
        "teknofest-experiment-tracker.py. Bu dosyalar, k\u00f6k dizinde "
        "da\u011f\u0131n\u0131k \u015fekilde bulunuyordu ve mod\u00fcler bir yap\u0131ya sahip de\u011fildi."
    )
    pdf.body_text("Tespit edilen sorunlar:")
    pdf.bullet_point("Konfig\u00fcrasyon de\u011ferleri kodun i\u00e7ine g\u00f6m\u00fcl\u00fc (hardcoded)")
    pdf.bullet_point("Test, e\u011fitim ve de\u011ferlendirme kodlar\u0131 i\u00e7 i\u00e7e ge\u00e7mi\u015f")
    pdf.bullet_point("Yeniden kullan\u0131labilirlik (reusability) \u00e7ok d\u00fc\u015f\u00fck")
    pdf.bullet_point("requirements.txt, Dockerfile ve .gitignore eksik")
    pdf.bullet_point("Deney takibi (experiment tracking) yap\u0131land\u0131r\u0131lmam\u0131\u015f")

    # \u2500\u2500 3. REFACTORING \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("3. Yaz\u0131l\u0131m Refactoring S\u00fcreci")
    pdf.section_title("3.1 Dosya Yap\u0131s\u0131 D\u00f6n\u00fc\u015f\u00fcm\u00fc")
    pdf.body_text(
        "Da\u011f\u0131n\u0131k haldeki t\u00fcm script dosyalar\u0131, end\u00fcstriyel standartlara uygun "
        "bir src/ dizin yap\u0131s\u0131na ta\u015f\u0131nd\u0131. Hedeflenen mimari \u015f\u00f6yledir:"
    )
    pdf.code_block(
        "\u00d6NCEK\u0130 YAPI:\n"
        "  teknofest-benchmark-runner.py  (1002 satir, monolitik)\n"
        "  teknofest-experiment-tracker.py (monolitik)\n"
        "  src/ (bos alt dizinler)\n"
        "\n"
        "SONRAK\u0130 YAPI:\n"
        "  configs/base_config.yaml\n"
        "  configs/model/distilberturk.yaml\n"
        "  src/data/preprocessor.py\n"
        "  src/features/brand_features.py\n"
        "  src/models/cross_encoder.py\n"
        "  src/training/distillation.py\n"
        "  src/experiment/config_loader.py\n"
        "  src/experiment/tracker.py\n"
        "  scripts/run_experiment.py\n"
        "  scripts/run_benchmark.py\n"
        "  scripts/run_dashboard.py"
    )

    pdf.section_title("3.2 Konfig\u00fcrasyon Sistemi")
    pdf.body_text(
        "T\u00fcm deney parametreleri (model tipi, \u00f6\u011frenme oran\u0131, batch boyutu, "
        "distillation alfa de\u011feri, threshold) YAML dosyalar\u0131na \u00e7\u0131kar\u0131ld\u0131. "
        "Bu sayede bir deney ba\u015flatmak i\u00e7in sadece 'python "
        "scripts/run_experiment.py --config configs/base_config.yaml' "
        "komutunu \u00e7al\u0131\u015ft\u0131rmak yeterli hale geldi."
    )

    pdf.section_title("3.3 MLOps Entegrasyonu")
    pdf.body_text(
        "MLflow tabanl\u0131 deney takip sistemi kuruldu. Her deney otomatik "
        "olarak parametreleri, metrikleri ve model a\u011f\u0131rl\u0131klar\u0131n\u0131 kaydeder. "
        "Bu sayede y\u00fczlerce deney aras\u0131nda kolayca kar\u015f\u0131la\u015ft\u0131rma yapmak "
        "m\u00fcmk\u00fcnd\u00fcr."
    )

    # \u2500\u2500 4. MET\u0130N \u00d6N \u0130\u015eLEME \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("4. Metin \u00d6n \u0130\u015fleme Mod\u00fcl\u00fc")
    pdf.body_text(
        "T\u00fcrk\u00e7e e-ticaret metinleri i\u00e7in \u00f6zel bir \u00f6n i\u015fleme mod\u00fcl\u00fc "
        "geli\u015ftirildi (src/data/preprocessor.py). Bu mod\u00fcl \u015funlar\u0131 yapar:"
    )
    pdf.bullet_point(
        "T\u00fcrk\u00e7e \u00d6zel K\u00fc\u00e7\u00fck Harf: Python'\u0131n varsay\u0131lan lower() fonksiyonu "
        "T\u00fcrk\u00e7e \u0130/i ve I/\u0131 karakterlerini do\u011fru d\u00f6n\u00fc\u015ft\u00fcremez. \u00d6zel bir "
        "_turkish_lower() fonksiyonu yaz\u0131ld\u0131."
    )
    pdf.bullet_point(
        "Yaz\u0131m Hatas\u0131 D\u00fczeltme: SymSpell k\u00fct\u00fcphanesi ile edit distance "
        "tabanl\u0131 h\u0131zl\u0131 yaz\u0131m d\u00fczeltme. E-ticaret \u00f6zel s\u00f6zl\u00fck deste\u011fi."
    )
    pdf.bullet_point(
        "Lemmatizasyon: Zeyrek (Zemberek Python portu) ile morfolojik "
        "analiz. 'ayakkab\u0131lar' \u2192 'ayakkab\u0131' gibi d\u00f6n\u00fc\u015f\u00fcmler."
    )

    # \u2500\u2500 5. \u00d6ZELL\u0130K \u00c7IKARIMI \u2500\u2500
    pdf.chapter_title("5. \u00d6zellik \u00c7\u0131kar\u0131m\u0131 (Feature Engineering)")
    pdf.body_text(
        "Modelin ba\u015far\u0131s\u0131, ham metin d\u0131\u015f\u0131nda ek \u00f6zellikler kullan\u0131lmas\u0131na "
        "ba\u011fl\u0131d\u0131r. Olu\u015fturulan \u00f6zellikler \u015funlard\u0131r:"
    )
    pdf.bullet_point(
        "Fuzzy Brand Match Score: RapidFuzz k\u00fct\u00fcphanesi ile sorgu ve marka "
        "aras\u0131ndaki karakter benzerlik oran\u0131 (0\u2013100 float)."
    )
    pdf.bullet_point(
        "Has Brand Match: Fuzzy skorun 85 e\u015fi\u011fini a\u015f\u0131p a\u015fmad\u0131\u011f\u0131n\u0131 g\u00f6steren "
        "binary (0/1) g\u00f6sterge."
    )

    # \u2500\u2500 6. MODEL E\u011e\u0130T\u0130M\u0130 \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("6. Model E\u011fitim Stratejisi")
    pdf.section_title("6.1 Cross-Encoder")
    pdf.body_text(
        "DistilBERTurk tabanl\u0131 Cross-Encoder modeli, sorgu-\u00fcr\u00fcn \u00e7iftlerini "
        "birle\u015ftirerek 3 s\u0131n\u0131fl\u0131 (Alakas\u0131z / K\u0131smen Alakal\u0131 / \u00c7ok Alakal\u0131) "
        "s\u0131n\u0131fland\u0131rma yapar. Batch boyutu 32, max_length 128 token olarak "
        "ayarlanm\u0131\u015ft\u0131r."
    )

    pdf.section_title("6.2 Teacher-Student Distillation")
    pdf.body_text(
        "B\u00fcy\u00fck BGE-M3 modeli (Teacher) taraf\u0131ndan \u00fcretilen soft-label'lar, "
        "DistilBERTurk (Student) modeline Margin-MSE loss fonksiyonu ile "
        "aktar\u0131l\u0131r. Margin-MSE, klasik MSE'den farkl\u0131 olarak pairwise "
        "ranking sinyali ta\u015f\u0131r: iki \u00fcr\u00fcn aras\u0131ndaki skor fark\u0131n\u0131n (margin) "
        "Teacher ve Student modellerinde tutarl\u0131 olmas\u0131n\u0131 hedefler."
    )
    pdf.body_text(
        "Loss = \u03b1 \u00d7 CE_Loss + (1 \u2212 \u03b1) \u00d7 MarginMSE_Loss\n"
        "\u03b1 = 0.7 (base_config.yaml'dan okunur)"
    )

    # \u2500\u2500 7. WEB ARAY\u00dcZ\u00dc \u2500\u2500
    pdf.chapter_title("7. Web Aray\u00fcz\u00fc Geli\u015ftirme")
    pdf.body_text(
        "Projenin sunumu i\u00e7in Next.js (App Router) tabanl\u0131, Brutalist "
        "ve F\u00fct\u00fcristik bir web aray\u00fcz\u00fc geli\u015ftirildi. Kullan\u0131lan "
        "teknolojiler:"
    )
    pdf.bullet_point("Next.js 15 \u2014 React Server Components")
    pdf.bullet_point("Three.js / @react-three/fiber \u2014 3D veri sim\u00fclasyonu")
    pdf.bullet_point("Framer Motion \u2014 Mikro animasyonlar")
    pdf.bullet_point("GSAP \u2014 Scroll-triggered parallax efektleri")
    pdf.bullet_point("Zustand \u2014 Teknik/Basit mod state y\u00f6netimi")
    pdf.body_text(
        "Aray\u00fcz, kullan\u0131c\u0131y\u0131 bir 'terminal boot sim\u00fclasyonu' ile kar\u015f\u0131lar, "
        "ard\u0131ndan e-ticaretin veri kaosunu 3D par\u00e7ac\u0131k animasyonuyla "
        "g\u00f6rselle\u015ftirerek, \u00e7\u00f6z\u00fcm\u00fcn nas\u0131l \u00e7al\u0131\u015ft\u0131\u011f\u0131n\u0131 interaktif mod\u00fcllerde "
        "anlat\u0131r. Sa\u011f \u00fcstteki 'Teknik/Basit' toggle ile t\u00fcm i\u00e7erik an\u0131nda "
        "de\u011fi\u015fir."
    )

    # \u2500\u2500 8. DA\u011eITIM \u2500\u2500
    pdf.add_page()
    pdf.chapter_title("8. Da\u011f\u0131t\u0131m ve DevOps")
    pdf.body_text(
        "Proje, Docker tabanl\u0131 offline da\u011f\u0131t\u0131m i\u00e7in yap\u0131land\u0131r\u0131lm\u0131\u015ft\u0131r. "
        "Dockerfile, t\u00fcm ba\u011f\u0131ml\u0131l\u0131klar\u0131 ve model a\u011f\u0131rl\u0131klar\u0131n\u0131 i\u00e7erir. "
        "FastAPI \u00fczerinden REST API sunulur."
    )
    pdf.code_block(
        "# Docker ile calistirma\n"
        "docker build -t deep-pipeline .\n"
        "docker run -p 8000:8000 \\\n"
        "  -v ./local_model:/app/local_model \\\n"
        "  deep-pipeline"
    )

    # \u2500\u2500 9. SONU\u00c7 \u2500\u2500
    pdf.chapter_title("9. Sonu\u00e7 ve Gelecek Ad\u0131mlar")
    pdf.body_text(
        "Deep-Pipeline, da\u011f\u0131n\u0131k bir prototipten profesyonel bir MLOps "
        "pipeline'\u0131na ba\u015far\u0131yla d\u00f6n\u00fc\u015ft\u00fcr\u00fclm\u00fc\u015ft\u00fcr. Konfig\u00fcrasyon odakl\u0131 "
        "mimari sayesinde h\u0131zl\u0131 deney d\u00f6ng\u00fcs\u00fc, Docker ile offline da\u011f\u0131t\u0131m, "
        "ve XAI dashboard ile \u015feffaf karar verme mekanizmas\u0131 "
        "sa\u011flanm\u0131\u015ft\u0131r."
    )
    pdf.body_text(
        "Gelecek ad\u0131mlar:\n"
        "\u2022 Yar\u0131\u015fma veri seti a\u00e7\u0131kland\u0131\u011f\u0131nda baseline e\u011fitimi\n"
        "\u2022 Multi-Seed Ensemble optimizasyonu\n"
        "\u2022 ONNX Runtime ile CPU inference optimizasyonu\n"
        "\u2022 Streamlit dashboard'\u0131n ger\u00e7ek verilerle test edilmesi"
    )

    # \u2500\u2500 KAYDET \u2500\u2500
    out_path = os.path.join(OUTPUT_DIR, "Teknofest_Gelistirme_Raporu.pdf")
    pdf.output(out_path)
    print(f"  [OK] Gelistirme Surec Raporu olusturuldu: {out_path}")
    return out_path


if __name__ == "__main__":
    print("=" * 60)
    print("  TEKNOFEST 2026 \u2014 PDF Rapor \u00dcretici")
    print("=" * 60)
    print()

    path1 = generate_system_design_report()
    path2 = generate_development_report()

    print()
    print("=" * 60)
    print(f"  T\u00fcm raporlar ba\u015far\u0131yla olu\u015fturuldu!")
    print(f"  Konum: {OUTPUT_DIR}")
    print("=" * 60)
