import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import base64
import os

st.title("Détection & Classification : Poubelle Pleine / Vide")
st.write("Uploader une image pour détecter la poubelle et son état.")

MODEL_PATH = "best.pt"

# Charger le modèle
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

# Upload image
img_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])

# Convert OpenCV → PIL
def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Bouton Télécharger le modèle
def download_button(file_path, label):
    with open(file_path, "rb") as f:
        file_data = f.read()
        b64 = base64.b64encode(file_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_path}">{label}</a>'
        st.markdown(href, unsafe_allow_html=True)

if img_file is not None:
    image = Image.open(img_file).convert("RGB")     # ⚠ assure 3 channels (évite erreur 4 channels PNG)
    image_np = np.array(image)

    st.image(image, caption="Image uploadée", use_container_width=True)

    # Prédiction
    try:
        results = model.predict(image_np)
    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
        st.stop()

    # Image avec bounding-box
    result_img = results[0].plot()
    st.image(result_img, caption="Résultat de la détection", use_container_width=True)

    # Lire le résultat
    if len(results[0].boxes) > 0:
        cls = int(results[0].boxes.cls[0])
        conf = float(results[0].boxes.conf[0])

        if cls == 0:
            st.success(f"🟢 Poubelle **PLEINE** (confiance : {conf:.2f})")
        elif cls == 1:
            st.success(f"🟡 Poubelle **VIDE** (confiance : {conf:.2f})")
        else:
            st.warning("Classe inconnue détectée 🤔")
    else:
        st.error("❌ Aucune poubelle détectée.")

# Télécharger le modèle
st.subheader("Télécharger votre modèle YOLO")
if os.path.exists(MODEL_PATH):
    download_button(MODEL_PATH, "Télécharger best.pt")
else:
    st.warning("⚠ Modèle introuvable dans le repo GitHub.")
