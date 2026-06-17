import streamlit as st
import cv2
import numpy as np

# --- CẤU HÌNH TRANG & BỘ TOẠ ĐỘ CỐ ĐỊNH MỚI CỦA BẠN ---
st.set_page_config(page_title="AI Canteen - Test Khung Cắt Mới", layout="wide")

# Cập nhật bộ tọa độ mới bạn vừa căn chỉnh thành công
BOX_COORDS = {
    "O_Canh": {"x": 0.16, "y": 0.1,  "w": 0.33, "h": 0.49}, 
    "O_Com":  {"x": 0.56, "y": 0.12, "w": 0.25, "h": 0.47}, 
    "O_M1":   {"x": 0.18, "y": 0.61, "w": 0.2,  "h": 0.31}, 
    "O_M2":   {"x": 0.39, "y": 0.62, "w": 0.2, "h": 0.3}, 
    "O_M3":   {"x": 0.60, "y": 0.62, "w": 0.19, "h": 0.31}
}

st.title("🍱 Kiểm Tra Vị Trí Khung Cắt Mới (640x480)")
st.info("Hệ thống đang khóa kích thước ảnh ở độ phân giải chuẩn. Vui lòng bấm chụp để xem bộ tọa độ mới đã ôm khít các ngăn chưa.")

camera_photo = st.camera_input("📸 Hãy đưa khay cơm vào và bấm chụp")

if camera_photo:
    # 1. Đọc dữ liệu ảnh đầu vào từ camera
    file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 🔥 2. ÉP KÍCH THƯỚC: Chuẩn hóa về 640x480 để khớp hoàn toàn với tọa độ bạn vừa chỉnh
    img_rgb = cv2.resize(img_rgb, (640, 480))
    img_debug = img_rgb.copy()
    
    # 3. Tính toán pixel và vẽ các khung chữ nhật lên hình
    for name, box in BOX_COORDS.items():
        x = int(box['x'] * 640)
        y = int(box['y'] * 480)
        w = int(box['w'] * 640)
        h = int(box['h'] * 480)
        
        # Vẽ hình chữ nhật màu xanh lá dày 3px quanh các ngăn thức ăn
        cv2.rectangle(img_debug, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Ghi nhãn tên ô ngay góc trên
        cv2.putText(img_debug, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- HIỂN THỊ KẾT QUẢ ---
    st.markdown("---")
    col_main, col_crops = st.columns([1.6, 1])
    
    with col_main:
        st.subheader("🖼️ Khung hình tổng quan (Tỷ lệ thực tế)")
        st.image(img_debug, use_container_width=True, caption="Ảnh sau khi ép kích thước và vẽ khung bằng tọa độ mới")
        
    with col_crops:
        st.subheader("✂️ Các vùng ảnh sẽ được gửi vào AI")
        # Cắt thử từng ô nhỏ để bạn kiểm tra xem thức ăn có bị mất góc không
        for name, box in BOX_COORDS.items():
            x = int(box['x'] * 640)
            y = int(box['y'] * 480)
            w = int(box['w'] * 640)
            h = int(box['h'] * 480)
            
            cropped = img_rgb[y:y+h, x:x+w]
            if cropped.size > 0:
                st.image(cropped, width=130, caption=f"Ảnh cắt thực tế: {name}")
    
