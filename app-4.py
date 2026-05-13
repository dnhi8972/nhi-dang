import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="Stress Check AI", page_icon="🧠")

# Tiêu đề "xịn" như yêu cầu trước đó
st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 22px; color: #666; margin-bottom: 0px;">Hệ thống phân tích tâm lý AI</p>
        <h1 style="font-size: 60px; color: #8E44AD; font-family: 'Arial Black'; margin-top: -10px;">
            STRESS CHECK
        </h1>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Hàm tải các file mô hình
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
    st.error("⚠️ Không tìm thấy file mô hình! Hãy kiểm tra lại GitHub của bạn.")
    st.stop()

# Giao diện nhập liệu trực quan
st.subheader("📊 Nhập thông số sinh hoạt hàng ngày")
col1, col2 = st.columns(2)

with col1:
    work = st.slider("Giờ làm việc mỗi ngày", 0.0, 16.0, 8.0)
    sleep = st.slider("Giờ ngủ mỗi đêm", 0.0, 12.0, 7.0)
    coffee = st.number_input("Số ly cà phê đã uống", 0, 15, 2)

with col2:
    age = st.number_input("Độ tuổi của bạn", 15, 100, 25)
    activity = st.number_input("Phút vận động thể chất", 0, 300, 30)

# Nút dự đoán
if st.button("PHÂN TÍCH TÌNH TRẠNG STRESS", use_container_width=True):
    # Chuẩn bị dữ liệu
    input_data = np.array([[work, sleep, coffee, age, activity]])
    input_scaled = scaler.transform(input_data)
    
    # Dự đoán
    prediction_idx = model.predict(input_scaled)[0]
    result = le.inverse_transform([prediction_idx])[0]
    
    st.markdown("---")
    st.write("### 🔍 Kết quả phân tích từ AI:")
    
    # Hiển thị theo màu sắc hợp lý
    if result == "Normal":
        st.success(f"🌟 Tình trạng: **{result.upper()} (Bình thường)**")
        st.info("Tuyệt vời! Lối sống của bạn đang rất cân bằng. Hãy duy trì nhé!")
    elif result == "Warning":
        st.warning(f"⚠️ Tình trạng: **{result.upper()} (Cảnh báo)**")
        st.info("Bạn đang có dấu hiệu căng thẳng. Hãy dành thời gian nghỉ ngơi và bớt cà phê lại.")
    else:
        st.error(f"🚨 Tình trạng: **{result.upper()} (Căng thẳng cao)**")
        st.info("CẢNH BÁO: Mức độ stress của bạn đang ở mức báo động. Cần thư giãn ngay lập tức!")
