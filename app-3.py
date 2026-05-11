import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="PriorityCheck AI", page_icon="🚨")

# Tiêu đề thiết kế theo yêu cầu
st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 22px; color: #666; margin-bottom: 0px;">Hệ thống phân tích Logistics</p>
        <h1 style="font-size: 60px; color: #1E88E5; font-family: 'Arial Black'; margin-top: -10px;">
            PRIORITY AI
        </h1>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        with open('logistics_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('logistics_columns.pkl', 'rb') as f: cols = pickle.load(f)
        return model, scaler, cols
    except: return None, None, None

model, scaler, cols = load_assets()

if model is None:
    st.error("❌ Không tìm thấy các file model trên GitHub. Vui lòng upload lại!")
    st.stop()

# Giao diện nhập liệu
st.write("### 📝 Nhập thông số vận hành")
input_values = []
col1, col2 = st.columns(2)

for i, col_name in enumerate(cols):
    with col1 if i % 2 == 0 else col2:
        val = st.number_input(f"Chỉ số {col_name}", value=0.0, step=0.1)
        input_values.append(val)

if st.button("PHÂN TÍCH MỨC ĐỘ ƯU TIÊN", use_container_width=True):
    # 1. Chuẩn hóa dữ liệu đầu vào
    input_array = np.array([input_values])
    input_scaled = scaler.transform(input_array)
    
    # 2. Lấy xác suất (Probability)
    # Đây là điểm mấu chốt để dự đoán "hợp lý"
    prob = model.predict_proba(input_scaled)[0][1] * 100 
    
    st.markdown("---")
    st.write(f"#### Kết quả phân tích xác suất: **{prob:.1f}%**")
    
    # 3. Logic phân cấp mức độ ưu tiên
    if prob < 35:
        st.success("🟢 **MỨC ĐỘ ƯU TIÊN: THẤP**")
        st.info("Hệ thống đánh giá tình trạng ổn định. Có thể xử lý theo quy trình thông thường.")
    elif 35 <= prob < 75:
        st.warning("🟡 **MỨC ĐỘ ƯU TIÊN: TRUNG BÌNH**")
        st.info("Cần bắt đầu chú ý. Đơn hàng/Công việc có dấu hiệu cần được điều phối sớm.")
    else:
        st.error("🔴 **MỨC ĐỘ ƯU TIÊN: CAO (KHẨN CẤP)**")
        st.info("CẢNH BÁO: Cần xử lý ngay lập tức để tránh gây tắc nghẽn hệ thống Logistics!")

    # Hiển thị thanh tiến trình trực quan
    st.progress(prob / 100)
