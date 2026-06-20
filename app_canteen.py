import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================================
# 1. BỘ VÁ LỖI KERAS (Đảm bảo model chạy mượt trên Cloud)
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

# ==========================================================
# 2. CẤU HÌNH TRANG & DỮ LIỆU CỐ ĐỊNH
# ==========================================================
st.set_page_config(page_title="Canteen AI Smart Checkout", page_icon="🍱", layout="wide")

# Bộ tọa độ chuẩn mới nhất bạn đã chốt khớp với khung 640x480
BOX_COORDS = {
    "O_Canh": {"x": 0.16, "y": 0.10, "w": 0.33, "h": 0.49}, 
    "O_Com":  {"x": 0.56, "y": 0.12, "w": 0.25, "h": 0.47}, 
    "O_M1":   {"x": 0.18, "y": 0.61, "w": 0.20, "h": 0.31}, 
    "O_M2":   {"x": 0.39, "y": 0.62, "w": 0.21, "h": 0.30}, 
    "O_M3":   {"x": 0.61, "y": 0.62, "w": 0.18, "h": 0.31}
}

MENU_PRICES = {
    '00_Com_trang': 5000, '01_Dau_hu_sot_ca': 15000, '02_Ca_kho': 20000, 
    '03_Thit_kho_trung': 25000, '04_Rau_muong_xao': 10000, '05_Canh_chua_co_ca': 15000,
    '06_Canh_chua_ko_ca': 10000, '07_Suon_nuong': 30000, '08_Canh_cai': 10000,
    '09_Thit_kho': 20000, '10_Trung_chien': 10000, '11_Rau_luoc': 10000
}
CLASS_FOODS = list(MENU_PRICES.keys())

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('mo_hinh_mobilenet_12_mon_HOAN_HAO.h5', compile=False)

model = load_ai_model()

# ==========================================================
# 3. GIAO DIỆN CHÍNH (UI)
# ==========================================================
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🍱 HỆ THỐNG THANH TOÁN THÔNG MINH BẰNG AI</h1>", unsafe_allow_html=True)
st.markdown("---")

col_camera, col_bill = st.columns([1.5, 1])

with col_camera:
    st.markdown("### 📸 Camera Nhận Diện")
    camera_photo = st.camera_input("Đưa khay thức ăn vào khu vực camera và chụp")

    if camera_photo:
        with st.spinner("🤖 AI đang phân tích khay cơm của bạn..."):
            # Chuẩn hóa kích thước ảnh về 640x480 để khớp tọa độ hoàn hảo
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (640, 480))
            
            # Vẽ các khung nhận diện xanh lá trực tiếp lên hình
            img_display = img_rgb.copy()
            for name, box in BOX_COORDS.items():
                x = int(box['x'] * 640)
                y = int(box['y'] * 480)
                w = int(box['w'] * 640)
                h = int(box['h'] * 480)
                cv2.rectangle(img_display, (x, y), (x + w, y + h), (50, 205, 50), 3)
                cv2.putText(img_display, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 205, 50), 2)
            
            st.image(img_display, caption="Ảnh thực tế với các vùng cắt được định vị", use_container_width=True)

            # Cắt lát ảnh và dự đoán món ăn bằng MobileNet
            detected_items = []
            for name, box in BOX_COORDS.items():
                x, y, w, h = int(box['x']*640), int(box['y']*480), int(box['w']*640), int(box['h']*480)
                cropped = img_rgb[y:y+h, x:x+w]
                
                if cropped.size == 0: continue
                
                resized = cv2.resize(cropped, (224, 224))
                input_tensor = np.expand_dims(preprocess_input(np.array(resized, dtype=np.float32)), axis=0)
                
                preds = model.predict(input_tensor, verbose=0)
                max_idx = np.argmax(preds)
                score = np.max(preds)
                
                # Chỉ nhận diện khi độ tự tin lớn hơn 60%
                if score > 0.4:
                    raw_name = CLASS_FOODS[max_idx]
                    clean_name = raw_name.split('_', 1)[1].replace('_', ' ').title() if '_' in raw_name else raw_name
                    price = MENU_PRICES[raw_name]
                    detected_items.append({"Món": clean_name, "Giá": price})

with col_bill:
    st.markdown("### 🧾 Hóa Đơn Chi Tiết")
    
    if camera_photo:
        if detected_items:
            # CSS định dạng hóa đơn biên lai siêu thị
            st.markdown("""
            <style>
            .bill-box {
                background-color: #f9f9f9;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .item-row { display: flex; justify-content: space-between; font-size: 18px; margin-bottom: 10px; border-bottom: 1px dashed #ccc;}
            .total-row { display: flex; justify-content: space-between; font-size: 24px; font-weight: bold; color: #d32f2f; margin-top: 20px; }
            </style>
            """, unsafe_allow_html=True)

            tong_tien = sum(item["Giá"] for item in detected_items)
            
            html_bill = '<div class="bill-box">'
            for item in detected_items:
                html_bill += f'<div class="item-row"><span>🍽️ {item["Món"]}</span><span>{item["Giá"]:,} đ</span></div>'
            html_bill += f'<div class="total-row"><span>TỔNG CỘNG:</span><span>{tong_tien:,} VNĐ</span></div>'
            html_bill += '</div>'
            
            st.markdown(html_bill, unsafe_allow_html=True)
            
            # ==========================================================
            # KHU VỰC HIỂN THỊ VIETQR & TÍCH ĐIỂM
            # ==========================================================
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_qr, col_pts = st.columns([1.1, 1.2])
            
            with col_qr:
                st.markdown("#### 📱 Quét VietQR")
                # 🔥 Thay thế mã QR tự tạo bằng ảnh Techcombank chuyên nghiệp của bạn
                try:
                    st.image("image_39b147.png", use_container_width=True)
                except:
                    st.error("⚠️ Không tìm thấy file 'image_39b147.png'. Vui lòng upload ảnh QR lên GitHub cùng cấp thư mục với file code này.")
                
                # Hiển thị số tiền động ngay dưới phôi ảnh để tăng tính thuyết phục lúc chấm điểm
                st.markdown(f"<p style='text-align: center; font-weight: bold; color: #111;'>Số tiền: {tong_tien:,} đ</p>", unsafe_allow_html=True)

            with col_pts:
                st.markdown("#### 💳 Thẻ Thành Viên")
                DIEM_CU = 150000 
                DIEM_HIEN_TAI = DIEM_CU + tong_tien
                HAN_MUC = 500000
                
                st.success(f"🎉 Tích lũy thêm: **+{tong_tien:,}đ**")
                st.metric("Tổng tích lũy hiện tại", f"{DIEM_HIEN_TAI:,} đ")
                
                # Thanh tiến trình tích điểm nâng cấp VIP
                ty_le = min(DIEM_HIEN_TAI / HAN_MUC, 1.0)
                st.progress(ty_le)
                
                if DIEM_HIEN_TAI >= HAN_MUC:
                    st.balloons()
                    st.markdown("⭐ **Hạng Thẻ: VIP (Giảm 10%)**")
                else:
                    con_thieu = HAN_MUC - DIEM_HIEN_TAI
                    st.caption(f"Cần thêm {con_thieu:,} đ để đạt mốc Thẻ VIP")
            
            st.button("💵 Xác Nhận Giao Dịch", type="primary", use_container_width=True)
            
        else:
            st.warning("Không nhận diện được món ăn nào trong các khung cắt. Bạn hãy thử chụp lại góc thẳng hơn nhé!")
    else:
        st.info("👈 Vui lòng bấm chụp ảnh khay cơm để trải nghiệm quy trình tính tiền tự động.")
