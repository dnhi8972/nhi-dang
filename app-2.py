import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cấu hình trang web
st.set_page_config(page_title="Định giá xe máy AI", page_icon="🏍️", layout="centered")

# Hàm tải các file mô hình (Dùng cache để app chạy mượt hơn)
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
    except Exception as e:
        return None, None, None

# Tải dữ liệu
model, model_columns, brand_map = load_assets()

# Giao diện chính
if model is None:
    st.error("⚠️ Không tìm thấy các file dữ liệu (.pkl) trên GitHub. Hãy đảm bảo bạn đã upload: moto_model.pkl, columns.pkl, và brand_model_map.pkl")
else:
    st.title("🏍️ Dự báo giá xe máy cũ (AI)")
    st.write("Nhập thông tin xe để nhận giá gợi ý từ hệ thống dữ liệu thị trường.")
    st.divider()

    # Layout chia làm 2 cột
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Thông tin cơ bản")
        hang = st.selectbox("Chọn hãng xe", list(brand_map.keys()))
        
        # Loại xe tự động cập nhật theo hãng
        danh_sach_xe = brand_map[hang]
        loai_xe = st.selectbox("Chọn dòng xe", danh_sach_xe)
        
        doi_xe = st.number_input("Năm sản xuất (Đời)", 2010, 2025, 2023)

    with col2:
        st.subheader("Tình trạng thực tế")
        so_km = st.number_input("Số KM đã đi", 0, 500000, 5000, step=500)
        
        # Tình trạng xe - Đây là phần quan trọng nhất để thay đổi giá
        tinh_trang = st.selectbox("Tình trạng ngoại quan", [
            "Mới (Đẹp, nguyên zin)", 
            "Cũ (Trầy xước nhẹ)", 
            "Cũ (Trầy nhiều, hao mòn)"
        ])

    # Nút bấm dự đoán
    if st.button("XEM GIÁ DỰ BÁO", use_container_width=True):
        # 1. Tính toán tuổi xe
        tuoi_xe = 2025 - doi_xe
        
        # 2. Tạo DataFrame rỗng với đúng các cột lúc huấn luyện
        # Khởi tạo tất cả bằng 0
        X_input = pd.DataFrame(columns=model_columns)
        X_input.loc[0] = 0
        
        # 3. Điền các giá trị số
        if 'Số KM' in X_input.columns:
            X_input['Số KM'] = so_km
        if 'Tuổi xe' in X_input.columns:
            X_input['Tuổi xe'] = tuoi_xe
            
        # 4. Kích hoạt (gán = 1) cho các cột phân loại (One-hot encoding)
        # Tìm và gán 1 cho Hãng
        col_hang = f"Hãng_{hang}"
        if col_hang in X_input.columns:
            X_input[col_hang] = 1
            
        # Tìm và gán 1 cho Loại xe
        col_loai = f"Loại xe_{loai_xe}"
        if col_loai in X_input.columns:
            X_input[col_loai] = 1
            
        # Tìm và gán 1 cho Tình trạng (Khớp chính xác với cột 'Tình trạng chuẩn' khi train)
        col_tinh_trang = f"Tình trạng chuẩn_{tinh_trang}"
        if col_tinh_trang in X_input.columns:
            X_input[col_tinh_trang] = 1

        # 5. Thực hiện dự báo
        try:
            du_bao = model.predict(X_input)[0]
            
            # Giới hạn giá không bị âm
            gia_cuoi = max(du_bao, 1000000)
            
            # Hiển thị kết quả
            st.divider()
            st.balloons()
            st.
