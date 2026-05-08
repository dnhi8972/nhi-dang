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

model, model_columns, brand_map = load_assets()

st.title("🏍️ Hệ thống định giá xe máy cũ")

col1, col2 = st.columns(2)

with col1:
    hang = st.selectbox("Chọn hãng xe", list(brand_map.keys()))
    loai_xe = st.selectbox("Chọn dòng xe", brand_map[hang])
    doi_xe = st.number_input("Năm sản xuất", 2010, 2025, 2023)

with col2:
    so_km = st.number_input("Số KM đã đi", 0, 500000, 5000)
    # CẬP NHẬT LẠI 3 TÙY CHỌN KHỚP VỚI MODEL MỚI
    tinh_trang = st.selectbox("Tình trạng xe", [
        "Mới (Đẹp, nguyên zin)", 
        "Cũ (Trầy xước nhẹ)", 
        "Cũ (Trầy nhiều, hao mòn)"
    ])

if st.button("Dự đoán giá ngay"):
    tuoi_xe = 2025 - doi_xe
    
    input_df = pd.DataFrame({
        'Số KM': [so_km],
        'Tuổi xe': [tuoi_xe],
        'Hãng': [hang],
        'Loại xe': [loai_xe],
        'Tình trạng chuẩn': [tinh_trang]  # SỬA LẠI TÊN CỘT Ở ĐÂY
    })
    
    input_encoded = pd.get_dummies(input_df)
    
    final_input = pd.DataFrame(columns=model_columns).fillna(0)
    final_input = pd.concat([final_input, input_encoded], axis=0).fillna(0)
    final_input = final_input[model_columns]
    
    res = model.predict(final_input)[0]
    
    st.divider()
    st.success(f"### Giá dự báo: {res:,.0f} VNĐ")
