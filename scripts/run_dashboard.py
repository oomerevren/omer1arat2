import os
import sys
import requests
import streamlit as st
import pandas as pd
import time

API_URL = os.environ.get("DEEP_PIPELINE_API", "http://localhost:8000")

st.set_page_config(page_title="Deep-Pipeline XAI Dashboard", page_icon="🔍", layout="wide")

st.title("🔍 Teknofest 2026: Açıklanabilir Yapay Zeka (XAI) Dashboard")
st.markdown("Model kararlarını özellik skorları ve metin açıklamalarıyla analiz eder.")


def api_available() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_resource
def get_api_status():
    return api_available()


api_ok = get_api_status()
if api_ok:
    st.success(f"API bağlantısı aktif: {API_URL}")
else:
    st.warning(
        f"API erişilemiyor ({API_URL}). "
        "Başlatmak için: `uvicorn src.deployment.api:app --port 8000`"
    )

st.sidebar.header("Deney Ayarları")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

query = st.text_input("Arama Sorgusu:", "siyah deri erkek cüzdan")
product = st.text_input("Ürün Başlığı:", "Derimod Hakiki Deri Cüzdan Siyah")
brand = st.text_input("Marka:", "Derimod")
category = st.text_input("Kategori:", "Aksesuar > Cüzdan")

if st.button("Analiz Et"):
    payload = {
        "search_query": query,
        "product_name": product,
        "brand": brand,
        "category": category,
    }

    with st.spinner("Model analiz ediyor..."):
        if api_ok:
            try:
                resp = requests.post(f"{API_URL}/explain", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"API hatası: {e}")
                st.stop()
        else:
            # Offline fallback — doğrudan model yükle
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from src.experiment.config_loader import load_config
            from src.models.cross_encoder import CrossEncoderModel
            from src.xai.explainer import ModelExplainer

            config = load_config("configs/base_config.yaml")
            model_path = "./experiments/outputs/baseline/cross_encoder"
            if os.path.exists(model_path):
                model = CrossEncoderModel.load(model_path)
            else:
                ce = config.get("model", {}).get("cross_encoder", {})
                model = CrossEncoderModel(model_name=ce.get("model_name", "dbmdz/distilbert-base-turkish-cased"))
            explainer = ModelExplainer(model, config)
            data = explainer.explain(query, product, brand=brand, category=category)

    pred = data["prediction"]
    st.success("Analiz Tamamlandı!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Tahmin", pred.get("predicted_class", "?"))
    col2.metric("Güven", f"{pred.get('confidence', 0)*100:.1f}%")
    col3.metric(
        "Karar",
        "EŞLEŞTİ" if pred.get("confidence", 0) >= confidence_threshold else "ALAKASIZ",
    )

    st.subheader("Açıklama")
    st.info(data.get("summary_tr", ""))

    st.subheader("Özellik Skorları")
    features = data.get("features", {})
    if features:
        feat_df = pd.DataFrame([
            {"Özellik": k, "Skor": round(v, 4)} for k, v in features.items()
        ])
        st.bar_chart(feat_df.set_index("Özellik"))

    st.subheader("Detaylı Özellik Açıklamaları")
    for exp in data.get("feature_explanations", []):
        icon = {"positive": "✅", "negative": "⚠️", "neutral": "ℹ️"}.get(exp.get("impact"), "ℹ️")
        st.write(f"{icon} **{exp.get('label')}**: {exp.get('value')} — {exp.get('detail', '')}")

    if data.get("visual", {}).get("class_probabilities"):
        st.subheader("Sınıf Olasılıkları")
        prob_df = pd.DataFrame(data["visual"]["class_probabilities"])
        st.bar_chart(prob_df.set_index("class"))
