import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Hệ thống Phân loại AI", page_icon="🎯")

# Giao diện tiêu đề
st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 20px; color: #555;">Ứng dụng Trí tuệ nhân tạo</p>
        <h1 style="font-size: 50px; color: #007BFF; font-family: sans-serif;">AI CLASSIFIER</h1>
    </div>
    """, unsafe_allow_html=True)

# Load model và danh sách cột
@st.cache_resource
def load_model():
    try:
        with open('logistics_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('logistics_columns.pkl', 'rb') as f:
            cols = pickle.load(f)
        return model, cols
    except:
        return None, None

model, cols = load_model()

if model is None:
    st.error("⚠️ Thiếu file model trên GitHub! Hãy upload logistics_model.pkl và logistics_columns.pkl")
    st.stop()

st.info("Nhập các thông số đầu vào để AI dự đoán kết quả:")

# Tạo các ô nhập liệu tự động dựa trên tên cột
input_data = {}
col1, col2 = st.columns(2)

for i, col_name in enumerate(cols):
    with col1 if i % 2 == 0 else col2:
        input_data[col_name] = st.number_input(f"Nhập {col_name}", value=0.0)

# Nút dự đoán
if st.button("DỰ ĐOÁN KẾT QUẢ", use_container_width=True):
    # Chuyển dữ liệu nhập thành DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Dự đoán
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input).max() * 100

    st.write("---")
    if prediction == 1:
        st.success(f"### KẾT QUẢ: LOẠI 1 (Đạt/Thành công)")
    else:
        st.warning(f"### KẾT QUẢ: LOẠI 0 (Không đạt/Thất bại)")
        
    st.write(f"Độ tin cậy của AI: **{probability:.2f}%**")
