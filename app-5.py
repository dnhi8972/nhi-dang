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

shape_labels = {
    0: "Hình Tròn 🔴",
    1: "Hình Tam Giác 🔺",
    2: "Hình Vuông 🔲",
    3: "Hình Chữ Nhật █",
    4: "Hình Thang ▰",
    5: "Hình Bình Hành ▱"
}

st.subheader("✍️ Hãy vẽ một hình học xuống bảng dưới đây:")
st.caption("💡 Mẹo: Vẽ hình to rõ ràng ở chính giữa bảng để AI dễ nhận diện chính xác nhất nhé!")

# Tạo bảng vẽ Real-time với nét cọ tối ưu (stroke_width=14)
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  
    stroke_width=14,                      
    stroke_color="#FFFFFF",               
    background_color="#000000",           
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Kiểm tra xem có nét vẽ nào trên bảng chưa
    if np.sum(canvas_result.image_data[:, :, 0]) > 0:
        
        # Xử lý đưa nét vẽ về ma trận 28x28 chuẩn hóa giống lúc train
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img_resized = img.resize((28, 28))
        img_array = np.array(img_resized) / 255.0
        
        # Dự đoán kết quả
        flat_data = img_array.flatten().reshape(1, -1)
        pred_class = model.predict(flat_data)[0]
        result_text = shape_labels[pred_class]
        
        st.markdown("---")
        st.write("### 🔍 Kết quả nhận diện thời gian thực:")
        st.info(f"AI dự đoán đây là: **{result_text}**")
    else:
        st.write("👉 *Hãy đặt chuột xuống bảng và vẽ một hình nào đó nha!*")
