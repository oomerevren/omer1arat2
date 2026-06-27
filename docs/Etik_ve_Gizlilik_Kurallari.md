# Etik ve Gizlilik Kuralları

Bu belge, Teknofest E-Ticaret Hackathonu kapsamında geliştirilen Deep-Pipeline projesinin veri gizliliği, etik kuralları ve adil yapay zeka prensiplerini tanımlar.

## 1. Veri Gizliliği (Data Privacy)
- **Kişisel Verilerin Korunması**: Proje kapsamında işlenen veriler e-ticaret platformlarının arama sorgularını ve ürün açıklamalarını içerir. Bu verilerde kişiyi tanımlayabilecek (PII - Personally Identifiable Information) isim, adres, TC Kimlik numarası gibi bilgiler bulunması halinde, bu veriler regex ve kural tabanlı yöntemlerle maskelenir.
- **Veri Saklama ve İmha**: Eğitim verileri sadece şifrelenmiş lokal disklerde veya kapalı devre sunucularda tutulur. Yarışma bitiminde, platform kuralları gereği ham veriler güvenli şekilde silinecektir.

## 2. Adalet ve Önyargı (Fairness & Bias)
- **Marka ve Ürün Kayırmacılığı**: Model (Cross-Encoder), belirli büyük markaları diğerlerine karşı haksız şekilde öne çıkarmaz. Fuzzy matching ve semantik benzerlik algoritmaları objektif kriterlere (metin örtüşmesi, anlamsal bağ) dayanır.
- **Dil ve İfade Önyargısı**: Türkçe doğal dil işleme aşamasında bölgesel ağızlar, lehçeler veya yazım hataları cezalandırılmaz. SymSpell typo correction modülü sayesinde kullanıcının ifade biçiminden bağımsız, adil sonuçlar üretilir.

## 3. Şeffaflık ve Açıklanabilirlik (Transparency)
- **Açıklanabilir Yapay Zeka (XAI)**: Sistem "kara kutu" (black box) değildir. Her bir tahminin nedeni (hangi tokenların etkili olduğu, hangi feature'ların skoru yükselttiği) LIME/SHAP ve Attention ağırlıkları aracılığıyla kullanıcılara veya denetçilere sunulur (Streamlit Dashboard üzerinden).

## 4. Kaynak Tüketimi ve Çevresel Etki (Green AI)
- Model mimarimiz, devasa LLM'ler yerine optimize edilmiş DistilBERTurk (66M parametre) ve INT8/ONNX tabanlı kuantizasyon kullanarak minimum enerji tüketimi hedefler.
