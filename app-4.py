import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Stress Check AI", page_icon="🧠")

st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 22px; color: #666; margin-bottom: 0px;">Hệ thống phân tích tâm lý AI</p>
        <h1 style="font-size: 60px; color: #8E44AD; font-family: 'Arial Black'; margin-top: -10px;">
            STRESS CHECK
        </h1>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

@st.cache_resource
def load_assets():
    try:
        with open('stress_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('stress_scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('stress_le.pkl', 'rb') as f: le = pickle.load(f)
        with open('stress_columns.pkl', 'rb') as f: cols = pickle.load(f)
        return model, scaler, le, cols
    except:
        return None, None, None, None

model, scaler, le, cols = load_assets()

if model is None:
    st.error("⚠️ Không tìm thấy file mô hình! Hãy kiểm tra lại GitHub.")
    st.stop()

st.subheader("📊 Nhập thông số sinh hoạt hằng ngày của bạn ")
col1, col2 = st.columns(2)

with col1:
    work = st.number_input("Giờ làm việc (VD: 11.1)", min_value=0.0, max_value=24.0, value=8.0, step=0.1)
    sleep = st.number_input("Giờ ngủ (VD: 4.5)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    coffee = st.number_input("Số ly cà phê", min_value=0, max_value=20, value=2, step=1)

with col2:
    age = st.number_input("Độ tuổi", min_value=10, max_value=100, value=25, step=1)
    activity = st.number_input("Phút vận động", min_value=0, max_value=500, value=30, step=1)

if st.button("PHÂN TÍCH TÌNH TRẠNG STRESS", use_container_width=True):
    # Chuẩn bị dữ liệu
    input_data = np.array([[work, sleep, coffee, age, activity]])
    input_scaled = scaler.transform(input_data)
    
    # Dự đoán
    prediction_idx = model.predict(input_scaled)[0]
    result = le.inverse_transform([prediction_idx])[0]
    
    st.markdown("---")
    st.write("### 🔍 Kết quả phân tích từ AI:")
    
    if result == "Normal":
        st.success(f"🌟 Tình trạng: **{result.upper()} (Bình thường)**")
    elif result == "Warning":
        st.warning(f"⚠️ Tình trạng: **{result.upper()} (Cảnh báo)**")
    else:
        st.error(f"🚨 Tình trạng: **{result.upper()} (Căng thẳng cao)**")
