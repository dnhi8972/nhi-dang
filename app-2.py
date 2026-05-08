import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Định giá xe máy AI", page_icon="🏍️")

@st.cache_resource
def load_assets():
    with open('moto_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    with open('brand_model_map.pkl', 'rb') as f:
        brand_map = pickle.load(f)
    return model, model_columns, brand_map

try:
    model, model_columns, brand_map = load_assets()
except:
    st.error("⚠️ Không tìm thấy các file dữ liệu (.pkl). Hãy chạy code train trên Colab trước!")
    st.stop()

st.title("🏍️ Hệ thống định giá xe máy cũ")
st.write("Dự báo giá dựa trên dữ liệu thị trường mới nhất")

col1, col2 = st.columns(2)

with col1:
    hang = st.selectbox("Chọn hãng xe", list(brand_map.keys()))
    # Loại xe sẽ thay đổi theo Hãng đã chọn
    loai_xe = st.selectbox("Chọn dòng xe", brand_map[hang])
    doi_xe = st.number_input("Năm sản xuất (Đời xe)", 2010, 2025, 2022)

with col2:
    so_km = st.number_input("Số KM đã đi", 0, 500000, 10000)
    # Lấy danh sách tình trạng mẫu (hoặc bạn có thể tự nhập list)
    tinh_trang = st.selectbox("Tình trạng xe", ["Mới - Không trầy xước", "Cũ - Trầy xước nhẹ", "Cũ - Hư hỏng"])

if st.button("Dự đoán giá ngay"):
    tuoi_xe = 2025 - doi_xe
    
    # Tạo input chuẩn
    input_df = pd.DataFrame({
        'Số KM': [so_km],
        'Tuổi xe': [tuoi_xe],
        'Hãng': [hang],
        'Loại xe': [loai_xe],
        'Tình trạng xe (Cũ/Mới - Trầy xước/Hư hỏng)': [tinh_trang]
    })
    
    # One-hot encoding
    input_encoded = pd.get_dummies(input_data) # Lỗi nhẹ ở đây, sửa thành:
    input_encoded = pd.get_dummies(input_df)
    
    # Khớp cột
    final_input = pd.DataFrame(columns=model_columns).fillna(0)
    final_input = pd.concat([final_input, input_encoded], axis=0).fillna(0)
    final_input = final_input[model_columns]
    
    # Dự đoán
    res = model.predict(final_input)[0]
    
    st.divider()
    st.subheader(f"Giá dự báo: :green[{res:,.0f} VNĐ]")
    st.info("Mức giá này dựa trên các dòng xe tương tự trong hệ thống dữ liệu của bạn.")
