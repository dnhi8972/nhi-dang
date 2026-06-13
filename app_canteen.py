import streamlit as st
import cv2
import numpy as np
import qrcode
from PIL import Image
import tensorflow as tf
import io
import json

# ==========================================================
# 🔥 BỘ VÁ LỖI CẤU HÌNH KERAS TỰ ĐỘNG
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

fix_streamlit_keras_bug()
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# CẤU HÌNH TRANG WEB & BIẾN TOÀN CỤC
# ==========================================
st.set_page_config(page_title="AI Canteen Smart Checkout", page_icon="🍱", layout="wide")

if 'customer_db' not in st.session_state:
    st.session_state.customer_db = {"0912345678": 450}

# Khởi tạo toạ độ TƯƠNG ĐỐI (0.0 -> 1.0) vào bộ nhớ tạm
if 'box_coords' not in st.session_state:
    st.session_state.box_coords = {
        "O_Man1 (Top-Left)":    {"x": 0.05, "y": 0.05, "w": 0.40, "h": 0.40},
        "O_Man2 (Top-Right)":   {"x": 0.55, "y": 0.05, "w": 0.40, "h": 0.40},
        "O_Com (Bottom-Left)":  {"x": 0.05, "y": 0.50, "w": 0.40, "h": 0.40},
        "O_Canh (Bottom-Right)":{"x": 0.55, "y": 0.50, "w": 0.40, "h": 0.40}
    }

MENU_PRICES = {
    '00_Com_trang': 5000,       '01_Dau_hu_sot_ca': 15000, '02_Ca_kho': 20000, 
    '03_Thit_kho_trung': 25000, '04_Rau_muong_xao': 10000, '05_Canh_chua_co_ca': 15000,
    '06_Canh_chua_ko_ca': 10000,'07_Suon_nuong': 30000,    '08_Canh_cai': 10000,
    '09_Thit_kho': 20000,       '10_Trung_chien': 10000,   '11_Rau_luoc': 10000
}
CLASS_FOODS = list(MENU_PRICES.keys())

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('mo_hinh_mobilenet_12_mon_HOAN_HAO.h5', compile=False)

# ==========================================
# CÁC HÀM XỬ LÝ ẢNH & DRAW BOX
# ==========================================
def draw_calibration_boxes(img_array, coords_dict):
    img_draw = img_array.copy()
    h_img, w_img, _ = img_draw.shape
    
    for name, box in coords_dict.items():
        # Chuyển đổi từ tỉ lệ % sang pixel thực tế của ảnh
        x = int(box['x'] * w_img)
        y = int(box['y'] * h_img)
        w = int(box['w'] * w_img)
        h = int(box['h'] * h_img)
        
        # Vẽ khung và tên
        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(img_draw, name.split(" ")[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img_draw

def process_and_predict(img_array, model, coords_dict):
    detected_items = []
    h_img, w_img, _ = img_array.shape
    
    for name, box in coords_dict.items():
        x = int(box['x'] * w_img)
        y = int(box['y'] * h_img)
        w = int(box['w'] * w_img)
        h = int(box['h'] * h_img)
        
        if y+h > h_img or x+w > w_img: continue
            
        cropped = img_array[y:y+h, x:x+w]
        if cropped.size == 0: continue
            
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
            detected_items.append({"Món": food_name, "Giá": price, "Ảnh": cropped, "Vị trí": name.split(" ")[0]})
            
    return detected_items

# ==========================================
# GIAO DIỆN CHÍNH (UI)
# ==========================================
st.title("🍱 Hệ Thống Nhận Diện Khay Cơm")

# Thanh Sidebar chứa công cụ
with st.sidebar:
    st.header("⚙️ Cài đặt hệ thống")
    che_do_can_chinh = st.checkbox("🛠 Bật chế độ Căn Chỉnh Khung", value=True)
    
    if che_do_can_chinh:
        st.markdown("---")
        st.subheader("Bảng Điều Khiển Toạ Độ")
        selected_box = st.selectbox("Chọn ô cần chỉnh:", list(st.session_state.box_coords.keys()))
        
        st.write(f"**Đang chỉnh:** `{selected_box}`")
        col_x, col_y = st.columns(2)
        st.session_state.box_coords[selected_box]['x'] = col_x.slider("Vị trí X (Trái/Phải)", 0.0, 1.0, st.session_state.box_coords[selected_box]['x'], 0.01)
        st.session_state.box_coords[selected_box]['y'] = col_y.slider("Vị trí Y (Lên/Xuống)", 0.0, 1.0, st.session_state.box_coords[selected_box]['y'], 0.01)
        
        col_w, col_h = st.columns(2)
        st.session_state.box_coords[selected_box]['w'] = col_w.slider("Chiều rộng (W)", 0.05, 1.0, st.session_state.box_coords[selected_box]['w'], 0.01)
        st.session_state.box_coords[selected_box]['h'] = col_h.slider("Chiều cao (H)", 0.05, 1.0, st.session_state.box_coords[selected_box]['h'], 0.01)
        
        st.markdown("---")
        st.success("Khi căn chỉnh xong, hãy copy bộ toạ độ này dán vào code để cố định:")
        st.code(json.dumps(st.session_state.box_coords, indent=4), language="json")

model = load_ai_model()

col1, col2 = st.columns([1.5, 1])

with col1:
    camera_photo = st.camera_input("📸 Đưa khay cơm vào và chụp")
    
    if camera_photo:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if che_do_can_chinh:
            # Hiển thị ảnh kèm khung xanh để người dùng kéo thả thanh trượt
            st.subheader("Góc nhìn Camera (Đang căn chỉnh)")
            img_with_boxes = draw_calibration_boxes(img_rgb, st.session_state.box_coords)
            st.image(img_with_boxes, use_container_width=True)
            st.info("👈 Hãy dùng thanh trượt ở menu bên trái để khớp các khung xanh vào đúng vị trí khay thức ăn.")
        else:
            # Chạy AI dự đoán thực tế
            with st.spinner("AI đang phân tích món ăn..."):
                items = process_and_predict(img_rgb, model, st.session_state.box_coords)
                
            st.success(f"Nhận diện thành công {len(items)} món ăn!")
            cols = st.columns(4)
            for idx, item in enumerate(items):
                with cols[idx % 4]:
                    st.image(item["Ảnh"], width=100)
                    st.caption(f"{item['Món']}")

with col2:
    if not che_do_can_chinh:
        st.subheader("Hóa Đơn")
        if camera_photo and 'items' in locals() and items:
            tong_tien = sum(item["Giá"] for item in items)
            for item in items:
                st.write(f"- {item['Món']}: `{item['Giá']:,.0f} đ`")
            st.markdown(f"### Tổng: {tong_tien:,.0f} đ")
    elif che_do_can_chinh:
        st.warning("Vui lòng tắt 'Chế độ Căn Chỉnh Khung' ở menu bên trái để hệ thống tiến hành nhận diện AI và tính tiền.")
