# Kaggle Notebook Oluşturma Rehberi

Bu rehber, Teknofest 2026 E-Ticaret Hackathonu ilk aşaması (Datathon) için Kaggle üzerinde çözümünüzü nasıl göndereceğinizi ve kodunuzun reproducible (tekrarlanabilir) olmasını nasıl sağlayacağınızı anlatır.

## 1. Notebook Kurulumu
1. Kaggle'a giriş yapın ve yarışma sayfasına gidin.
2. "Code" sekmesinden "New Notebook" butonuna tıklayın.
3. Sağ panelden ayarları yapılandırın:
   - **Accelerator:** T4 GPU x2 veya P100 (Eğitim için), Inference için GPU kapalı (CPU only) - Şartname gereği değerlendirme CPU ile de çalışabilmelidir ancak Kaggle Notebook tarafında süre limitleri dahilinde GPU ile eğitip inference yapabilirsiniz.
   - **Internet:** "Internet on" (Kütüphaneleri indirmek için).

## 2. Veri Setinin Eklenmesi
Sağ panelde "Data" sekmesinde "Add Data" diyerek yarışma verisini (26 Haziran'da iletilecek olan) notebook'unuza bağlayın. 

## 3. Kod Düzeni (Best Practices)
Yarışma jürisi (Şartname 4.1'e göre) ilk 20 takımın notebook'larını inceleyecek. Kodunuz okunaklı ve tutarlı olmalıdır.

Aşağıdaki yapı örnek alınabilir:

```python
# 1. Kütüphane Kurulumları
!pip install transformers rapidfuzz catboost optuna -q

# 2. İçe Aktarmalar ve Sabitler
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42
MAX_LEN = 256
MODEL_NAME = "dbmdz/distilbert-base-turkish-cased"

# 3. Veri Ön İşleme (Preprocessing)
def preprocess_text(text):
    # Projedeki lowercase, zeyrek vb. kodlarını buraya taşıyın
    pass

# 4. Veri Yükleme ve Hazırlık
train_df = pd.read_csv('/kaggle/input/teknofest2026/train.csv')
test_df = pd.read_csv('/kaggle/input/teknofest2026/test.csv')

# 5. Model Eğitimi (Opsiyonel - Sadece Inference da yapılabilir)
# Eğer eğitim lokalde yapıldıysa, eğitilmiş ağırlıkları "Add Data -> New Dataset" ile ekleyip buradan yükleyin.
```

## 4. Submission Üretimi
`scripts/kaggle_submission.py` dosyasındaki mantığı Kaggle'da uygulamalısınız:

```python
# 6. Tahmin ve Submission
model.eval()
preds = []

# DataLoader ile test seti üzerinde tahmin
# probs = ...
# threshold = 0.55 # Optuna ile bulduğunuz eşik

# preds = (probs[:, 1] >= threshold).astype(int)

submission = pd.DataFrame({
    'product_id': test_df['product_id'],
    'search_query': test_df['search_query'],
    'prediction': preds
})

submission.to_csv('submission.csv', index=False)
print("Submission hazır!")
```

## 5. Kaydetme ve Gönderme
1. Sağ üstten **Save Version** butonuna tıklayın.
2. "Save & Run All (Commit)" seçeneğini işaretleyin.
3. Çalışma bittiğinde notebook sayfasındaki "Output" sekmesinden `submission.csv` dosyasını bulup "Submit" butonuna basarak Kaggle'a gönderin.
