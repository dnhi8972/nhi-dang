import streamlit as st
import cv2
import numpy as np
import qrcode
from PIL import Image
import tensorflow as tf
import io

# ==========================================================
# 🔥 BỘ VÁ LỖI CẤU HÌNH KERAS TỰ ĐỘNG - SỬA TRIỆT ĐỂ LỖI QUÀI
# ==========================================================
def fix_streamlit_keras_bug():
    def strip_bad_keys(d):
        if isinstance(d, dict):
            d.pop('quantization_config', None)
            d.pop('optional', None)
            for v in d.values():
                strip_bad_keys(v)
        elif isinstance(d, list):
            for item in d:
                strip_bad_keys(item)
    
    try:
        import keras.src.legacy.saving.serialization as legacy_serialization
        orig_deserialize = legacy_serialization.deserialize_keras_object
        def patched_deserialize(identifier, *args, **kwargs):
            strip_bad_keys(identifier)
            return orig_deserialize(identifier, *args, **kwargs)
        legacy_serialization.deserialize_keras_object = patched_deserialize
    except Exception:
        pass

    try:
        import keras.saving.serialization_lib as serialization_lib
        orig_deserialize_lib = serialization_lib.deserialize_keras_object
        def patched_deserialize_lib(identifier, *args, **kwargs):
            strip_bad_keys(identifier)
            return orig_deserialize_lib(identifier, *args, **kwargs)
        serialization_lib.deserialize_keras_object = patched_deserialize_lib
    except Exception:
        pass

# Kích hoạt bộ lọc lỗi trước khi nạp Model
fix_streamlit_keras_bug()
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# CẤU HÌNH TRANG WEB & BIẾN TOÀN CỤC
# ==========================================
st.set_page_config(page_title="AI Canteen Smart Checkout", page_icon="🍱", layout="wide")

if 'customer_db' not in st.session_state:
    st.session_state.customer_db = {
        "0912345678": 450, 
        "0987654321": 120
    }

MENU_PRICES = {
    '00_Com_trang': 5000,       '01_Dau_hu_sot_ca': 15000, '02_Ca_kho': 20000, 
    '03_Thit_kho_trung': 25000, '04_Rau_muong_xao': 10000, '05_Canh_chua_co_ca': 15000,
    '06_Canh_chua_ko_ca': 10000,'07_Suon_nuong': 30000,    '08_Canh_cai': 10000,
    '09_Thit_kho': 20000,       '10_Trung_chien': 10000,   '11_Rau_luoc': 10000
}
CLASS_FOODS = list(MENU_PRICES.keys())

TRAY_COORDS = {
    "O_Com":  {"x": 50,  "y": 300, "w": 200, "h": 200},
    "O_Canh": {"x": 270, "y": 300, "w": 200, "h": 200},
    "O_Man1": {"x": 50,  "y": 50,  "w": 150, "h": 150},
    "O_Man2": {"x": 220, "y": 50,  "w": 150, "h": 150}
}

# ==========================================
# CÁC HÀM XỬ LÝ LÕI
# ==========================================
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('mo_hinh_mobilenet_12_mon_HOAN_HAO.h5', compile=False)

def generate_qr_code(amount):
    qr_data = f"Thanh toan Canteen: {amount:,.0f} VND\nSTK: 123456789 (MOCK)"
    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white")
    
    buf = io.BytesIO()
    img_qr.save(buf, format="PNG")
    return buf.getvalue()

def process_and_predict(img_array, model):
    detected_items = []
    
    for ten_o, toa_do in TRAY_COORDS.items():
        x, y, w, h = toa_do["x"], toa_do["y"], toa_do["w"], toa_do["h"]
        
        if y+h > img_array.shape[0] or x+w > img_array.shape[1]:
            continue
            
        cropped = img_array[y:y+h, x:x+w]
        
        resized = cv2.resize(cropped, (224, 224))
        photo_array = np.array(resized, dtype=np.float32)
        preprocessed = preprocess_input(photo_array)
        input_tensor = np.expand_dims(preprocessed, axis=0)
        
        preds = model.predict(input_tensor, verbose=0)
        max_idx = np.argmax(preds)
        score = np.max(preds)
        
        if score > 0.6: 
            food_name = CLASS_FOODS[max_idx]
            price = MENU_PRICES[food_name]
            detected_items.append({"Món": food_name, "Giá": price, "Ảnh": cropped})
            
    return detected_items

# ==========================================
# GIAO DIỆN CHÍNH (UI)
# ==========================================
st.title("🍱 Hệ Thống Nhận Diện Khay Cơm & Thanh Toán Thông Minh")
st.markdown("---")

model = load_ai_model()

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("1. Đặt khay cơm vào camera")
    camera_photo = st.camera_input("📸 Hãy đưa khay cơm vào khung hình và bấm nút Chụp")
    
    if camera_photo:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        with st.spinner("AI đang phân tích món ăn..."):
            items = process_and_predict(img_rgb, model)
            
        st.success(f"Nhận diện thành công {len(items)} món ăn!")
        
        cols = st.columns(4)
        for idx, item in enumerate(items):
            with cols[idx % 4]:
                st.image(item["Ảnh"], width=100)
                st.caption(f"{item['Món']}")

with col2:
    st.subheader("2. Hóa Đơn & Tích Điểm")
    
    if camera_photo and 'items' in locals() and items:
        tong_tien = sum(item["Giá"] for item in items)
        
        st.write("**Các món đã nhận diện:**")
        for item in items:
            st.write(f"- {item['Món']}: `{item['Giá']:,.0f} đ`")
            
        st.markdown(f"### Tổng tạm tính: {tong_tien:,.0f} đ")
        st.markdown("---")
        
        phone = st.text_input("Nhập SĐT Khách Hàng (Tích điểm):", placeholder="VD: 0912345678")
        giam_gia = 0
        diem_cong_them = int(tong_tien / 1000)
        
        if phone:
            diem_hien_tai = st.session_state.customer_db.get(phone, 0)
            st.info(f"Khách hàng đang có: **{diem_hien_tai} điểm**")
            
            if diem_hien_tai >= 500:
                giam_gia = int(tong_tien * 0.10)
                st.success("🎉 Khách hàng VIP: Được giảm 10%!")
                
                if st.button("Áp dụng giảm giá (-500 điểm)"):
                    st.session_state.customer_db[phone] -= 500
                    st.rerun()
            else:
                st.warning(f"Cần thêm {500 - diem_hien_tai} điểm để được giảm 10% tháng này.")
        
        thanh_tien = tong_tien - giam_gia
        
        st.markdown(f"## THÀNH TIỀN: {thanh_tien:,.0f} đ")
        if phone:
            st.caption(f"*(Sau khi thanh toán, SĐT {phone} sẽ được cộng thêm {diem_cong_them} điểm)*")
        
        st.markdown("---")
        st.write("Quét mã QR để thanh toán (Mock):")
        qr_bytes = generate_qr_code(thanh_tien)
        st.image(qr_bytes, width=200)
        
        if st.button("✅ HOÀN TẤT GIAO DỊCH", type="primary", use_container_width=True):
            if phone:
                st.session_state.customer_db[phone] = st.session_state.customer_db.get(phone, 0) + diem_cong_them
            st.balloons()
            st.success("Giao dịch thành công! Xin cảm ơn quý khách.")
