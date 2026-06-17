import streamlit as st
import cv2
import numpy as np

# --- CẤU HÌNH TRANG & TOẠ ĐỘ CỐ ĐỊNH CỦA BẠN ---
st.set_page_config(page_title="AI Canteen - Test Khung Cắt Standard", layout="wide")

BOX_COORDS = {
    "O_Canh (Top-Left)":      {"x": 0.11, "y": 0.09, "w": 0.40, "h": 0.48},
    "O_Com (Top-Right)":      {"x": 0.60, "y": 0.09, "w": 0.34, "h": 0.47},
    "O_Man1 (Bottom-Left)":   {"x": 0.12, "y": 0.59, "w": 0.25, "h": 0.31},
    "O_Man2 (Bottom-Center)": {"x": 0.39, "y": 0.59, "w": 0.25, "h": 0.31},
    "O_Man3 (Bottom-Right)":  {"x": 0.66, "y": 0.59, "w": 0.25, "h": 0.31}
}

st.title("🍱 Giao Diện Kiểm Tra Khung Cắt (Chuẩn Hóa Kích Thước)")
st.info("Hệ thống sẽ tự động đưa ảnh chụp về kích thước cố định 640x480 để đảm bảo tọa độ luôn luôn khớp.")

camera_photo = st.camera_input("📸 Hãy đưa khay cơm vào và chụp thử")

if camera_photo:
    # 1. Đọc ảnh từ Camera đầu vào
    file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 🔥 2. ÉP KÍCH THƯỚC ẢNH VỀ CHUẨN 640x480 (Đây là mấu chốt để không bị lệch tỷ lệ)
    img_rgb = cv2.resize(img_rgb, (640, 480))
    
    # 3. Tạo một bản sao để vẽ khung khoanh vùng lên ảnh
    img_debug = img_rgb.copy()
    
    # Tiến hành tính toán và vẽ các ô vuông dựa trên kích thước cố định 640x480
    for name, box in BOX_COORDS.items():
        x = int(box['x'] * 640)
        y = int(box['y'] * 480)
        w = int(box['w'] * 640)
        h = int(box['h'] * 480)
        
        # Vẽ hình chữ nhật màu xanh lá dày 3px quanh khay thức ăn
        cv2.rectangle(img_debug, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Ghi tên ngăn thức ăn ngay phía trên khung
        cv2.putText(img_debug, name.split(" ")[0], (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- HIỂN THỊ KẾT QUẢ TEST ---
    st.markdown("---")
    col_main, col_crops = st.columns([1.5, 1])
    
    with col_main:
        st.subheader("🖼️ Khung hình tổng quan")
        st.image(img_debug, use_container_width=True, caption="Ảnh thực tế máy tính nhận được sau khi ép kích thước (640x480)")
        
    with col_crops:
        st.subheader("✂️ Kết quả AI thực tế sẽ cắt ra")
        # Hiển thị riêng lẻ từng ngăn xem có bị mất góc hay dính viền không
        for name, box in BOX_COORDS.items():
            x = int(box['x'] * 640)
            y = int(box['y'] * 480)
            w = int(box['w'] * 640)
            h = int(box['h'] * 480)
            
            cropped = img_rgb[y:y+h, x:x+w]
            if cropped.size > 0:
                st.image(cropped, width=120, caption=f"{name.split(' ')[0]}")
