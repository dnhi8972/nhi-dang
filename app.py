import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cấu hình trang web
st.set_page_config(page_title="Định giá xe máy cũ", page_icon="🏍️")

# Load model và các cột đã lưu
@st.cache_resource
def load_assets():
    with open('moto_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, columns

try:
    model, model_columns = load_assets()
except:
    st.error("Không tìm thấy file model. Vui lòng kiểm tra lại!")

# GIAO DIỆN APP
st.title("🏍️ Dự đoán giá bán xe máy cũ")
st.write("Dựa trên dữ liệu thực tế và thuật toán Linear Regression")

# Nhập liệu
col1, col2 = st.columns(2)

with col1:
    hang = st.selectbox("Chọn hãng xe", ["Honda", "Yamaha", "Suzuki", "Piaggio", "SYM"])
    loai_xe = st.selectbox("Loại xe", ["Xe ga", "Xe số", "Xe côn tay"])
    doi_xe = st.number_input("Năm sản xuất (Đời xe)", 2000, 2024, 2020)

with col2:
    so_km = st.number_input("Số KM đã đi", 0, 500000, 10000)
    tinh_trang = st.selectbox("Tình trạng ngoại quan", 
                             ["Mới - Không trầy xước", "Cũ - Trầy xước nhẹ", "Cũ - Hư hỏng nhẹ"])

# XỬ LÝ DỮ LIỆU KHI NHẤN NÚT
if st.button("Dự đoán ngay"):
    # Tạo DataFrame từ input của người dùng
    tuoi_xe = 2024 - doi_xe
    data = {'Số KM': [so_km], 'Tuổi xe': [tuoi_xe]}
    
    # Tạo DataFrame tạm thời
    input_df = pd.DataFrame(data)
    
    # Tạo các cột dummy cho biến chữ
    user_input = pd.DataFrame([[hang, loai_xe, tinh_trang]], 
                             columns=['Hãng', 'Loại xe', 'Tình trạng xe (Cũ/Mới - Trầy xước/Hư hỏng)'])
    user_input_encoded = pd.get_dummies(user_input)
    
    # Gộp lại
    final_input = pd.concat([input_df, user_input_encoded], axis=1)
    
    # Đảm bảo input có đủ các cột như lúc train (điền 0 vào cột thiếu)
    for col in model_columns:
        if col not in final_input.columns:
            final_input[col] = 0
            
    final_input = final_input[model_columns] # Sắp xếp đúng thứ tự
    
    # Dự đoán
    prediction = model.predict(final_input)[0]
    
    # Hiển thị kết quả
    st.divider()
    if prediction < 0:
        st.warning("Thông tin xe không hợp lệ hoặc xe quá cũ để định giá.")
    else:
        st.balloons()
        st.subheader(f"Giá bán gợi ý: :green[{prediction:,.0f} VNĐ]")
        st.caption("Lưu ý: Giá trên chỉ mang tính chất tham khảo dựa trên mô hình toán học.")
