import streamlit as st
import cv2
import numpy as np

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Canteen - Bộ Chỉnh Tọa Độ Chuẩn", layout="wide")

st.title("⚙️ Công Cụ Căn Chỉnh Vùng Cắt (Tỷ Lệ Chuẩn Đã Khóa)")
st.info("Hệ thống đã khóa kích thước ảnh ở độ phân giải 640x480. Bạn hãy dùng các thanh trượt bên phải để kéo các ô xanh khớp với khay cơm của mình nhé.")

# Giá trị mặc định ban đầu dựa trên bộ tọa độ cũ của bạn
if 'coords' not in st.session_state:
    st.session_state.coords = {
        "O_Canh": {"x": 0.11, "y": 0.09, "w": 0.40, "h": 0.48},
        "O_Com":  {"x": 0.60, "y": 0.09, "w": 0.34, "h": 0.47},
        "O_M1":   {"x": 0.12, "y": 0.59, "w": 0.25, "h": 0.31},
        "O_M2":   {"x": 0.39, "y": 0.59, "w": 0.25, "h": 0.31},
        "O_M3":   {"x": 0.66, "y": 0.59, "w": 0.25, "h": 0.31}
    }

# Tạo giao diện chia làm 2 cột: Bên trái hiển thị hình, Bên phải chứa thanh trượt
col_img, col_ctrl = st.columns([1.5, 1])

# --- CỘT PHẢI: THANH TRƯỢT CĂN CHỈNH ---
with col_ctrl:
    st.subheader("🛠️ Chọn ô và kéo thanh trượt")
    
    # Cho người dùng chọn ô muốn sửa
    selected_box = st.selectbox("Chọn ô cần điều chỉnh:", list(st.session_state.coords.keys()))
    
    st.markdown(f"#### 📍 Đang chỉnh sửa: **{selected_box}**")
    
    # Các thanh trượt điều chỉnh tỉ lệ phần trăm từ 0.0 đến 1.0
    box = st.session_state.coords[selected_box]
    new_x = st.slider("Vị trí X (Trái/Phải)", 0.0, 1.0, float(box['x']), 0.01)
    new_y = st.slider("Vị trí Y (Trên/Dưới)", 0.0, 1.0, float(box['y']), 0.01)
    new_w = st.slider("Chiều rộng W", 0.0, 1.0, float(box['w']), 0.01)
    new_h = st.slider("Chiều cao H", 0.0, 1.0, float(box['h']), 0.01)
    
    # Cập nhật giá trị mới vào bộ nhớ tạm của app
    st.session_state.coords[selected_box] = {"x": new_x, "y": new_y, "w": new_w, "h": new_h}
    
    # Xuất ra code text để bạn copy khi đã chỉnh xong xuôi
    st.markdown("### 📋 Code tọa độ hiện tại của bạn:")
    st.code(str(st.session_state.coords).replace("'", '"'))

# --- CỘT TRÁI: HIỂN THỊ CAMERA VÀ VẼ KHUNG ---
with col_img:
    camera_photo = st.camera_input("📸 Hãy bấm chụp khay cơm")
    
    if camera_photo:
        # Đọc ảnh từ camera
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 🔥 KHÓA KÍCH THƯỚC: Ép ảnh về chuẩn 640x480 để đồng bộ tỷ lệ hiển thị
        img_rgb = cv2.resize(img_rgb, (640, 480))
        img_debug = img_rgb.copy()
        
        # Vẽ các khung lên hình dựa trên các thanh trượt đang kéo
        for name, b in st.session_state.coords.items():
            # Tính toán vị trí pixel trên nền ảnh kích thước cố định 640x480
            x_px = int(b['x'] * 640)
            y_px = int(b['y'] * 480)
            w_px = int(b['w'] * 640)
            h_px = int(b['h'] * 480)
            
            # Chọn màu đỏ cho ô đang chỉnh, màu xanh lá cho các ô còn lại
            color = (255, 0, 0) if name == selected_box else (0, 255, 0)
            thickness = 4 if name == selected_box else 2
            
            cv2.rectangle(img_debug, (x_px, y_px), (x_px + w_px, y_px + h_px), color, thickness)
            cv2.putText(img_debug, name, (x_px, y_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        st.image(img_debug, use_container_width=True, caption="Hình ảnh kiểm tra thực tế (Đã đưa về tỷ lệ 640x480)")
    
