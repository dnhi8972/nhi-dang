import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

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
st.set_page_config(page_title="AI Canteen Debug", layout="wide")

BOX_COORDS = {
    "O_Canh": {"x": 0.11, "y": 0.09, "w": 0.40, "h": 0.48},
    "O_Com":  {"x": 0.60, "y": 0.09, "w": 0.34, "h": 0.47},
    "O_M1":   {"x": 0.12, "y": 0.59, "w": 0.25, "h": 0.31},
    "O_M2":   {"x": 0.39, "y": 0.59, "w": 0.25, "h": 0.31},
    "O_M3":   {"x": 0.66, "y": 0.59, "w": 0.25, "h": 0.31}
}

MENU_PRICES = {'00_Com_trang': 5000, '01_Dau_hu_sot_ca': 15000, '02_Ca_kho': 20000, '03_Thit_kho_trung': 25000, '04_Rau_muong_xao': 10000, '05_Canh_chua_co_ca': 15000, '06_Canh_chua_ko_ca': 10000, '07_Suon_nuong': 30000, '08_Canh_cai': 10000, '09_Thit_kho': 20000, '10_Trung_chien': 10000, '11_Rau_luoc': 10000}
CLASS_FOODS = list(MENU_PRICES.keys())

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('mo_hinh_mobilenet_12_mon_HOAN_HAO.h5', compile=False)

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🍱 Kiểm Tra Cắt Khay (Debug Mode)")
camera_photo = st.camera_input("📸 Chụp khay cơm để test khung cắt")

if camera_photo:
    file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape
    
    # 🔥 VẼ KHUNG DEBUG LÊN ẢNH
    img_debug = img_rgb.copy()
    for name, box in BOX_COORDS.items():
        x, y, w, h = int(box['x']*w_img), int(box['y']*h_img), int(box['w']*w_img), int(box['h']*h_img)
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 5) # Khung xanh dày 5px
        cv2.putText(img_debug, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    st.subheader("Ảnh đã vẽ khung cắt (Đảm bảo các ô nằm đúng vị trí)")
    st.image(img_debug, use_container_width=True)
    
    # Hiển thị các ô đã cắt nhỏ (dùng để check xem AI cắt có bị mất góc không)
    st.subheader("Các ô đã được cắt ra:")
    cols = st.columns(5)
    for idx, (name, box) in enumerate(BOX_COORDS.items()):
        x, y, w, h = int(box['x']*w_img), int(box['y']*h_img), int(box['w']*w_img), int(box['h']*h_img)
        cropped = img_rgb[y:y+h, x:x+w]
        cols[idx].image(cropped, caption=f"Cắt: {name}")

    st.success("Nếu khung xanh đã bao quanh đúng 5 ngăn, bạn có thể yên tâm dùng toạ độ này!")
    
