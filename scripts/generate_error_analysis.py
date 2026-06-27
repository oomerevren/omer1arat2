import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def categorize_error(row):
    if row["true_label"] == 1 and row["pred_label"] == 0:
        return "False Negative (Missed Match)"
    elif row["true_label"] == 0 and row["pred_label"] == 1:
        return "False Positive (Spurious Match)"
    return "Correct"

def main():
    os.makedirs("reports", exist_ok=True)
    
    # Sentetik veri ile hata analizi oluşturma (gerçek veriye uygulanacak)
    np.random.seed(42)
    n_samples = 1000
    y_val = np.random.randint(0, 2, n_samples)
    val_preds = y_val.copy()
    
    # Biraz hata ekle
    flip_indices = np.random.choice(n_samples, size=150, replace=False)
    val_preds[flip_indices] = 1 - val_preds[flip_indices]
    
    val_df = pd.DataFrame({
        "search_term": ["sample query"] * n_samples,
        "product_name": ["sample product"] * n_samples,
        "true_label": y_val,
        "pred_label": val_preds
    })
    
    cm = confusion_matrix(y_val, val_preds)
    print("Confusion Matrix:\n", cm)
    
    errors = val_df[val_df["pred_label"] != val_df["true_label"]].copy()
    errors["error_type"] = errors.apply(categorize_error, axis=1)
    
    error_counts = errors["error_type"].value_counts()
    error_counts.to_csv("reports/error_categories.csv")
    print("Hata kategorileri kaydedildi: reports/error_categories.csv")

if __name__ == "__main__":
    main()
