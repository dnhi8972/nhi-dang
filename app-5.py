import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle
import numpy as np
from PIL import Image

st.set_page_config(page_title="Geometry Recognition AI", page_icon="📐")

st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 20px; color: #555; margin-bottom: 0px;">Hệ thống Nhận Diện Hình Học Real-time</p>
        <h1 style="font-size: 45px; color: #E67E22; font-family: 'Arial Black'; margin-top: -10px;">
            GEOMETRY PERCEPTRON
        </h1>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Tải mô hình đã train
@st.cache_resource
def load_model():
    try:
        with open('geometry_perceptron.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

if model is None:
    st.error("⚠️ Không tìm thấy file mô hình 'geometry_perceptron.pkl'. Vui lòng upload lên GitHub!")
    st.stop()

# Từ điển ánh xạ kết quả
shape_labels = {
    0: "Hình Tròn 🔴",
    1: "Hình Tam Giác 🔺",
    2: "Hình Vuông 🔲",
    3: "Hình Chữ Nhật █",
    4: "Hình Thang ▰",
    5: "Hình Bình Hành ▱"
}

st.subheader("✍️ Hãy vẽ một hình học xuống bảng dưới đây:")

# Tạo bảng vẽ Real-time
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # Không màu nền cho nét vẽ
    stroke_width=12,                      # Độ dày nét vẽ
    stroke_color="#FFFFFF",               # Nét vẽ màu trắng
    background_color="#000000",           # Bảng nền đen giống môi trường học của AI
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Xử lý hình ảnh khi người dùng vẽ
if canvas_result.image_data is not None:
    # Kiểm tra xem người dùng đã vẽ nét nào chưa (tránh nhận diện bảng trống)
    if np.sum(canvas_result.image_data[:, :, 0]) > 0:
        
        # Chuyển đổi ảnh từ bảng vẽ về dạng grayscale và resize thành 28x28
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img_resized = img.resize((28, 28))
        img_array = np.array(img_resized) / 255.0
        
        # Duỗi phẳng dữ liệu đầu vào thành 1 hàng (784 cột) đem đi dự đoán
        flat_data = img_array.flatten().reshape(1, -1)
        
        # Dự đoán kết quả
        pred_class = model.predict(flat_data)[0]
        result_text = shape_labels[pred_class]
        
        st.markdown("---")
        st.write("### 🔍 Kết quả nhận diện thời gian thực:")
        st.info(f"AI dự đoán đây là: **{result_text}**")
    else:
        st.write("👉 *Hãy đặt chuột xuống bảng và vẽ một hình nào đó nha!*")
