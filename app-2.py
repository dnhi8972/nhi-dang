import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Định giá xe máy AI", page_icon="🏍️")

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
    st.error("⚠️ Thiếu file .pkl trên GitHub. Hãy kiểm tra lại!")
    st.stop()

st.title("🏍️ Định giá xe máy AI")

# Giao diện nhập liệu
hang = st.selectbox("Hãng xe", list(brand_map.keys()))
loai_xe = st.selectbox("Dòng xe", brand_map[hang])
doi_xe = st.number_input("Năm sản xuất", 2010, 2025, 2023)
so_km = st.number_input("Số KM đã đi", 0, 500000, 5000)
tinh_trang = st.selectbox("Tình trạng", [
    "Mới (Đẹp, nguyên zin)", 
    "Cũ (Trầy xước nhẹ)", 
    "Cũ (Trầy nhiều, hao mòn)"
])

if st.button("Dự báo giá"):
    # 1. Chuẩn bị dữ liệu input (Toàn bộ là 0)
    X_input = pd.DataFrame(columns=model_columns)
    X_input.loc[0] = 0
    
    # 2. Điền các giá trị số
    if 'Số KM' in X_input.columns: X_input['Số KM'] = so_km
    if 'Tuổi xe' in X_input.columns: X_input['Tuổi xe'] = 2025 - doi_xe
    
    # 3. Gán giá trị 1 cho các cột chữ (CƠ CHẾ AN TOÀN)
    # Nếu không tìm thấy cột chính xác, nó sẽ không báo lỗi mà chỉ bỏ qua
    def safe_set(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in X_input.columns:
            X_input[col_name] = 1

    safe_set("Hãng", hang)
    safe_set("Loại xe", loai_xe)
    safe_set("Tình trạng chuẩn", tinh_trang)

    # 4. Dự đoán
    try:
        res = model.predict(X_input)[0]
        st.success(f"### Giá dự báo: {max(res, 1000000):,.0f} VNĐ")
    except Exception as e:
        st.error(f"Lỗi: {e}")
