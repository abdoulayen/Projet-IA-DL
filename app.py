import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import os

st.title("Détection d'image - YOLOv8")

# ---- Chargement du modèle ----
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---- Upload image ----
uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Image chargée", use_column_width=True)

    # ---- Inference ----
    st.write("🔍 **Prédiction en cours...**")

    results = model.predict(img, conf=0.4)

    # ---- Afficher le résultat ----
    result_img = results[0].plot()  # image annotée

    st.image(result_img, caption="Résultat", use_column_width=True)

    # ---- Afficher les labels détectés ----
    detections = results[0].boxes

    if len(detections) == 0:
        st.warning("Aucune classe détectée.")
    else:
        st.subheader("📌 Objets détectés :")
        for box in detections:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]
            st.write(f"- {class_name} ({confidence:.2f})")
