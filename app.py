import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import base64

# ----------------------------
# Parámetros
# ----------------------------
img_height, img_width = 224, 224  # usar los mismos del entrenamiento
class_names = ["Benigno", "Maligno"]

# ----------------------------
# Función para generar heatmap
# ----------------------------
def apply_heatmap_safe(img_array):
    img = (img_array * 255).astype('uint8')
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img_width, img_height))
    heatmap = heatmap.astype('float32') / 255.0
    return heatmap

# ----------------------------
# Funciones de preprocesado
# ----------------------------
def preprocess_image(image):
    img = np.array(image)
    img_resized = cv2.resize(img, (img_width, img_height))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    return img_array

def preprocess_image_heatmap(image):
    img_array = preprocess_image(image)
    heatmap_img = apply_heatmap_safe(img_array[0])
    heatmap_array = np.expand_dims(heatmap_img, axis=0)
    return heatmap_array

# ----------------------------
# Cargar modelo
# ----------------------------
model = tf.keras.models.load_model("modelo_cnn.h5")

# ----------------------------
# Cambiar fondo con CSS
# ----------------------------
def set_background_image(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background_image(r"C:\Users\aserr\Downloads\FondoWeb.jpg")

# ----------------------------
# Título
# ----------------------------
st.markdown("<h1 style='text-align:center; font-weight:bold; color:#FF0000;'>CLASIFICADOR DE MELANOMAS</h1>", unsafe_allow_html=True)
st.write("---")

# ----------------------------
# Subida de imagen
# ----------------------------
uploaded_file = st.file_uploader("📂 Suba la imagen que quiera analizar.", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Imagen cargada", use_container_width=True)
    
    # Preprocesar imagen normal
    img_array = preprocess_image(image)
    pred_normal = float(model.predict(img_array)[0][0])
    label_normal = class_names[int(pred_normal > 0.4)]

    # Preprocesar imagen con heatmap
    img_heatmap = preprocess_image_heatmap(image)
    pred_heatmap = float(model.predict(img_heatmap)[0][0])
    label_heatmap = class_names[int(pred_heatmap > 0.4)]
    
    # Mostrar resultados en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Predicción con imagen normal")
        st.progress(pred_normal)
        st.write(f"✅ Clase: **{label_normal}**")
        st.write(f"Probabilidad: {pred_normal:.2f}")
    
    with col2:
        st.markdown("### 🌡️ Predicción con imagen heatmap")
        st.image(img_heatmap[0], caption="Imagen heatmap", use_container_width=True)
        st.progress(pred_heatmap)
        st.write(f"✅ Clase: **{label_heatmap}**")
        st.write(f"Probabilidad: {pred_heatmap:.2f}")

    # Predicción combinada
    combined_prob = (pred_normal + pred_heatmap) / 2
    combined_label = class_names[int(combined_prob > 0.4)]
    st.write("---")
    st.markdown("### 🔹 Predicción combinada (normal + heatmap)")
    st.progress(combined_prob)
    st.write(f"✅ Clase final: **{combined_label}**")
    st.write(f"Probabilidad combinada: {combined_prob:.2f}")
