import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ----------------------------
# Parámetros
# ----------------------------
img_height, img_width = 224, 224  # usa los mismos del entrenamiento
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
# Función para preprocesar
# ----------------------------
def preprocess_image(image):
    img = np.array(image)
    img_resized = cv2.resize(img, (img_width, img_height))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    return img_array

# ----------------------------
# Función para preprocesar con heatmap
# ----------------------------
def preprocess_image_heatmap(image):
    img_array = preprocess_image(image)
    heatmap_img = apply_heatmap_safe(img_array[0])
    heatmap_array = np.expand_dims(heatmap_img, axis=0)
    return heatmap_array

# ----------------------------
# Cargar modelo
# ----------------------------
model = tf.keras.models.load_model("modelo_cnn.h5")

st.title("🔎 Clasificador de melanomas con CNN + Heatmap")

# ----------------------------
# Subir imagen
# ----------------------------
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Preprocesar imagen normal
    img_array = preprocess_image(image)
    pred_normal = model.predict(img_array)[0][0]
    label_normal = class_names[int(pred_normal > 0.4)]

    # Preprocesar imagen con heatmap
    img_heatmap = preprocess_image_heatmap(image)
    pred_heatmap = model.predict(img_heatmap)[0][0]
    label_heatmap = class_names[int(pred_heatmap > 0.4)]

    # Mostrar resultados
    st.write("### Predicción con imagen normal")
    st.write(f"✅ Clase: **{label_normal}**")
    st.write(f"Probabilidad: {pred_normal:.2f}")

    st.write("### Predicción con imagen en heatmap")
    st.image(img_heatmap[0])  # muestra la versión heatmap
    st.write(f"✅ Clase: **{label_heatmap}**")
    st.write(f"Probabilidad: {pred_heatmap:.2f}")

    # Opcional: combinación simple
    combined_prob = (pred_normal + pred_heatmap) / 2
    combined_label = class_names[int(combined_prob > 0.4)]
    st.write("### Predicción combinada (normal + heatmap)")
    st.write(f"✅ Clase final: **{combined_label}**")
    st.write(f"Probabilidad combinada: {combined_prob:.2f}")
