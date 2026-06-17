import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json

# ==========================================================
# 🔥 BỘ VÁ LỖI CẤU HÌNH KERAS
# ==========================================================
def fix_keras_error():
    try:
        import keras.src.legacy.saving.serialization as legacy_serialization
        def strip_keys(d):
            if isinstance(d, dict):
                d.pop('quantization_config', None)
                d.pop('optional', None)
                for v in d.values(): strip_keys(v)
            elif isinstance(d, list):
                for item in d: strip_keys(item)
        orig = legacy_serialization.deserialize_keras_object
        def patched(identifier, *args, **kwargs):
            strip_keys(identifier)
            return orig(identifier, *args, **kwargs)
        legacy_serialization.deserialize_keras_object = patched
    except: pass

fix_keras_error()
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# CẤU HÌNH & TOẠ ĐỘ CỐ ĐỊNH
# ==========================================
st.set_page_config(page_title="AI Canteen", layout="wide")

# Toạ độ 5 ngăn bạn đã chốt
BOX_COORDS = {
    "O_Canh (Top-Left)":      {"x": 0.11, "y": 0.09, "w": 0.40, "h": 0.48},
    "O_Com (Top-Right)":      {"x": 0.60, "y": 0.09, "w": 0.34, "h": 0.47},
    "O_Man1 (Bottom-Left)":   {"x": 0.12, "y": 0.59, "w": 0.25, "h": 0.31},
    "O_Man2 (Bottom-Center)": {"x": 0.39, "y": 0.59, "w": 0.25, "h": 0.31},
    "O_Man3 (Bottom-Right)":  {"x": 0.66, "y": 0.59, "w": 0.25, "h": 0.31}
}

MENU_PRICES = {
    '00_Com_trang': 10000, '01_Dau_hu_sot_ca': 25000, '02_Ca_kho': 30000, 
    '03_Thit_kho_trung': 30000, '04_Rau_muong_xao': 10000, '05_Canh_chua_co_ca': 25000,
    '06_Canh_chua_ko_ca': 10000, '07_Suon_nuong': 30000, '08_Canh_cai': 10000,
    '09_Thit_kho': 25000, '10_Trung_chien': 25000, '11_Rau_luoc': 10000
}
CLASS_FOODS = list(MENU_PRICES.keys())

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('mo_hinh_mobilenet_12_mon_HOAN_HAO.h5', compile=False)

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🍱 Hệ Thống Nhận Diện Khay Cơm Thông Minh")

col1, col2 = st.columns([1.5, 1])

with col1:
    camera_photo = st.camera_input("📸 Chụp khay cơm")
    model = load_ai_model()
    
    if camera_photo:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        img_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
        
        detected_items = []
        h_img, w_img, _ = img_rgb.shape
        
        # Nhận diện từng ô
        for name, box in BOX_COORDS.items():
            x, y, w, h = int(box['x']*w_img), int(box['y']*h_img), int(box['w']*w_img), int(box['h']*h_img)
            cropped = img_rgb[y:y+h, x:x+w]
            resized = cv2.resize(cropped, (224, 224))
            input_tensor = np.expand_dims(preprocess_input(np.array(resized, dtype=np.float32)), axis=0)
            
            preds = model.predict(input_tensor, verbose=0)
            if np.max(preds) > 0.6:
                food_name = CLASS_FOODS[np.argmax(preds)]
                detected_items.append({"Món": food_name, "Giá": MENU_PRICES[food_name]})

        st.success(f"Đã nhận diện {len(detected_items)} món ăn")

with col2:
    st.subheader("Hóa Đơn & Tích Điểm")
    if 'detected_items' in locals() and detected_items:
        tong_tien = sum(item["Giá"] for item in detected_items)
        for item in detected_items:
            st.write(f"- {item['Món']}: `{item['Giá']:,} đ`")
        st.markdown(f"### Tổng cộng: {tong_tien:,} đ")
        
        # 🔥 THANH TÍCH ĐIỂM
        DIEM_HIEN_TAI = tong_tien  # Bạn có thể cộng dồn từ database
        HAN_MUC_TOI_DA = 500000
        
        st.markdown(f"<h4 style='color: #2e7d32;'>{DIEM_HIEN_TAI:,} VND / {HAN_MUC_TOI_DA:,} VND</h4>", unsafe_allow_html=True)
        st.progress(min(DIEM_HIEN_TAI / HAN_MUC_TOI_DA, 1.0))
        
        if DIEM_HIEN_TAI >= HAN_MUC_TOI_DA:
            st.balloons()
            st.success("Đạt hạn mức tối đa!")
    
