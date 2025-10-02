import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import base64

# ----------------------------
# Parámetros
# ----------------------------
img_height, img_width = 224, 224
class_names = ["Benigno", "Maligno"]

# ----------------------------
# Funciones de preprocesado
# ----------------------------
def apply_heatmap_safe(img_array):
    img = (img_array * 255).astype('uint8')
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img_width, img_height))
    heatmap = heatmap.astype('float32') / 255.0
    return heatmap

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
# Fondo de la app
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
    .bubble {{
        background: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background_image(r"C:\Users\aserr\Downloads\FondoWeb.jpg")

# ----------------------------
# Título mejorado
# ----------------------------
st.markdown("""
<h1 style="
    text-align: center;
    font-weight: bold;
    font-size: 60px;
    color: #FFD700;
    text-shadow: 2px 2px 8px #000000;
    margin-bottom: 50px;
">
DETECTOR AVANZADO DE MELANOMAS
</h1>
""", unsafe_allow_html=True)

st.write("---")

# ----------------------------
# Subida de imagen
# ----------------------------
uploaded_file = st.file_uploader("Suba la imagen que quiera analizar.", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)
    
    # Preprocesar
    img_array = preprocess_image(image)
    pred_normal = float(model.predict(img_array)[0][0])
    label_normal = class_names[int(pred_normal > 0.4)]

    img_heatmap = preprocess_image_heatmap(image)
    pred_heatmap = float(model.predict(img_heatmap)[0][0])
    label_heatmap = class_names[int(pred_heatmap > 0.4)]

    # Columnas para mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="bubble">
        <h3>Predicción Imagen Normal</h3>
        <p>Clase: <b>{label_normal}</b></p>
        <p>Probabilidad: {pred_normal:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(pred_normal)

    with col2:
        st.markdown(f"""
        <div class="bubble">
        <h3> Predicción Imagen con mapa de calor</h3>
        <img src="data:image/png;base64,{base64.b64encode(cv2.imencode('.png', (img_heatmap[0]*255).astype(np.uint8))[1]).decode()}" style="width:100%;">
        <p> Clase: <b>{label_heatmap}</b></p>
        <p>Probabilidad: {pred_heatmap:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(pred_heatmap)

    # ----------------------------
    # Predicción combinada
    # ----------------------------
    combined_prob = (pred_normal + pred_heatmap) / 2
    combined_label = class_names[int(combined_prob > 0.4)]
    st.markdown(f"""
    <div class="bubble">
    <h3> Predicción Combinada</h3>
    <p>Clase final: <b>{combined_label}</b></p>
    <p>Probabilidad combinada: {combined_prob:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    st.progress(combined_prob)

    # ----------------------------
    # Mensaje interpretativo educado
    # ----------------------------
    if combined_prob <= 0.4:
        advice = "No parece haber indicios preocupantes. Mantén tu seguimiento habitual."
    elif combined_prob <= 0.6:
        advice = "Se recomienda consultar con un profesional para un examen más detallado."
    else:
        advice = "Hay indicios que podrían ser preocupantes. Se aconseja acudir al médico cuanto antes."

    st.markdown(f"""
    <div class="bubble">
    <h3>Interpretación</h3>
    <p>{advice}</p>
    </div>
    """, unsafe_allow_html=True)
