import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Motocheck - Dự đoán giá xe máy", page_icon="🏍️")

@st.cache_resource
def load_assets():
    try:
        with open('moto_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
        with open('brand_model_map.pkl', 'rb') as f:
            brand_map = pickle.load(f)
        return model, model_columns, brand_map
    except:
        return None, None, None

model, model_columns, brand_map = load_assets()

if model is None:
    st.error("⚠️ Không tìm thấy file dữ liệu trên GitHub!")
    st.stop()

# Tiêu đề thiết kế riêng
st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 24px; margin-bottom: 0px; color: #555; font-weight: bold;">
            Dự đoán giá bán xe máy cũ
        </p>
        <h1 style="font-size: 70px; margin-top: -10px; color: #FF4B4B; font-family: 'Arial Black', sans-serif; letter-spacing: -2px;">
            Motocheck
        </h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True) # Tạo một khoảng cách nhỏ bên dưới tiêu đề

col1, col2 = st.columns(2)

with col1:
    hang = st.selectbox("Chọn hãng xe", list(brand_map.keys()))
    loai_xe = st.selectbox("Chọn dòng xe", brand_map[hang])
    doi_xe = st.number_input("Năm sản xuất", 2010, 2025, 2023)

with col2:
    so_km = st.number_input("Số KM đã đi", 0, 500000, 5000)
    tinh_trang = st.selectbox("Tình trạng thực tế", [
        "Mới (Đẹp, nguyên zin)", 
        "Cũ (Trầy xước nhẹ)", 
        "Cũ (Trầy nhiều, hao mòn)"
    ])

if st.button("DỰ ĐOÁN GIÁ NGAY", use_container_width=True):
    # Tạo bảng input với toàn bộ cột đã học
    X_input = pd.DataFrame(columns=model_columns).fillna(0)
    X_input.loc[0] = 0
    
    # Điền số
    X_input['Số KM'] = so_km
    X_input['Tuổi xe'] = 2025 - doi_xe
    
    # Chuyển chữ thành số (Phải khớp 100% với lúc train)
    mapping = {
        "Mới (Đẹp, nguyên zin)": 3,
        "Cũ (Trầy xước nhẹ)": 2,
        "Cũ (Trầy nhiều, hao mòn)": 1
    }
    X_input['Tình trạng số'] = mapping[tinh_trang]
    
    # Điền Hãng và Loại xe
    def safe_set(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in X_input.columns:
            X_input[col_name] = 1

    safe_set("Hãng", hang)
    safe_set("Loại xe", loai_xe)

    # Dự đoán
    res = model.predict(X_input)[0]
    st.success(f"### Giá dự báo: {max(res, 1000000):,.0f} VNĐ")
