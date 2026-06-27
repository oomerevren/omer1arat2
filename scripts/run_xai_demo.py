import streamlit as st
import requests
import json

st.set_page_config(page_title="Deep-Pipeline XAI Demo", layout="wide")
st.title("📝 Deep-Pipeline — Açıklanabilir Ürün-Terim Eşleştirme")

col1, col2 = st.columns(2)
with col1:
    query = st.text_input("Arama sorgusu", value="kadın kışlık mont")
with col2:
    product = st.text_input("Ürün adı", value="Columbia Bayan Su Geçirmez Mont")

if st.button("Analiz Et"):
    try:
        response = requests.post("http://localhost:8000/explain", json={
            "search_query": query, 
            "product_name": product
        })
        if response.status_code == 200:
            data = response.json()
            st.subheader("📝 Token-Level Attention")
            st.write(data.get("attention", {}))

            st.subheader("📝 Feature Contributions (SHAP)")
            st.write(data.get("visual", {}).get("feature_bars", []))

            st.subheader("📝 Açıklama")
            st.info(data.get("summary_tr", "Açıklama bulunamadı."))
        else:
            st.error(f"API Hatası: {response.status_code}")
    except Exception as e:
        st.error(f"Bağlantı Hatası: {e}")
