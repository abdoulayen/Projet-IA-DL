import os
os.environ["YOLO_OFFLINE"] = "True"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import base64

st.title("Détection d’une poubelle (pleine ou vide)")
st.write("Uploader une image pour détecter la poubelle.")

# Charger le modèle ONNX (plus léger et compatible cloud)
@st.cache_resource
def load_model():
    try:
        return YOLO("best.onnx")
    except Exception as e:
        st.error(f"❌ Erreur model: {e}")
        return None

model = load_model()

def download_button(file_path, label):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<a href="data:application/octet-stream;base64,{b64}" download="{file_path}">{label}</a>',
            unsafe_allow_html=True
        )

uploaded_file = st.file_uploader("Uploader une image", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    st.image(image, caption="Image uploadée")

    with st.spinner("Analyse en cours..."):
        results = model.predict(
            img_np,
            imgsz=640,
            conf=0.25,
            max_det=1,
            device='cpu'
        )

    result_img = results[0].plot()
    st.image(result_img, caption="Résultat YOLO")

    if len(results[0].boxes) > 0:
        cls = int(results[0].boxes.cls[0])
        conf = float(results[0].boxes.conf[0])
        if cls == 0:
            st.success(f"🟢 Poubelle PLEINE (confiance {conf:.2f})")
        else:
            st.success(f"🟡 Poubelle VIDE (confiance {conf:.2f})")
    else:
        st.error("❌ Aucune poubelle détectée.")

st.subheader("Télécharger le modèle")
download_button("best.onnx", "Télécharger best.onnx")
