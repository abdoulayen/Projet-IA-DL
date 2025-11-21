import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import base64

# Configuration de la page
st.set_page_config(
    page_title="Détection de Poubelles",
    page_icon="🗑️",
    layout="wide"
)

# CSS personnalisé pour améliorer le design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .upload-section {
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        background-color: #f8fff8;
        margin: 20px 0;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: white;
    }
    .warning-box {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: white;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        backdrop-filter: blur(10px);
    }
    .download-btn {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 25px;
        cursor: pointer;
        font-weight: bold;
        text-decoration: none;
        display: inline-block;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Charger le modèle
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# Fonction pour créer un bouton de téléchargement stylisé
def styled_download_button(file_path, label):
    with open(file_path, "rb") as f:
        file_data = f.read()
        b64 = base64.b64encode(file_data).decode()
        href = f'''
        <a href="data:application/octet-stream;base64,{b64}" download="{file_path}" 
           class="download-btn">
           📥 {label}
        </a>
        '''
        st.markdown(href, unsafe_allow_html=True)

# Interface principale
def main():
    # Header
    st.markdown('<h1 class="main-header">🗑️ Détection Intelligente de Poubelles</h1>', unsafe_allow_html=True)
    st.markdown("### 🔍 Détection et classification des poubelles (pleines ou vides)")
    
    # Charger le modèle
    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return

    # Layout en colonnes
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Section upload
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📤 Uploader une image")
        st.markdown("**Formats supportés:** JPG, PNG, JPEG")
        img_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if img_file is not None:
            # Afficher l'image originale
            st.markdown("### 📷 Image originale")
            image = Image.open(img_file)
            
            # Convertir en RGB si nécessaire
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            st.image(image, use_container_width=True)

    with col2:
        if img_file is not None:
            st.markdown("### 🔍 Résultats")
            
            # Barre de progression
            progress_bar = st.progress(0)
            
            # Conversion et prédiction
            image_np = np.array(image)
            progress_bar.progress(30)
            
            with st.spinner("🔄 Analyse en cours..."):
                results = model.predict(image_np)
            progress_bar.progress(70)
            
            # Image avec bounding boxes
            result_img = results[0].plot()
            progress_bar.progress(90)
            
            st.image(result_img, caption="Image analysée", use_container_width=True)
            progress_bar.progress(100)
            
            # Résultats
            if len(results[0].boxes) > 0:
                cls = int(results[0].boxes.cls[0])
                conf = float(results[0].boxes.conf[0])
                
                # Affichage stylisé du résultat
                if cls == 0:
                    st.markdown(f'''
                    <div class="success-box">
                        <h3>🎉 Poubelle Détectée</h3>
                        <p style="font-size: 1.4rem; font-weight: bold;">🟢 État: PLEINE</p>
                        <p style="font-size: 1.1rem;">Confiance: {conf:.2%}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="warning-box">
                        <h3>🎉 Poubelle Détectée</h3>
                        <p style="font-size: 1.4rem; font-weight: bold;">🟡 État: VIDE</p>
                        <p style="font-size: 1.1rem;">Confiance: {conf:.2%}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Métriques
                st.markdown("### 📊 Métriques")
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Confiance", f"{conf:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_metric2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Objets détectés", len(results[0].boxes))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_metric3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    status = "Pleine" if cls == 0 else "Vide"
                    st.metric("État", status)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            else:
                st.error("❌ Aucune poubelle détectée dans l'image.")

    # Section téléchargement
    st.markdown("---")
    st.markdown("### 📁 Télécharger le modèle")
    styled_download_button("best.pt", "Télécharger le modèle YOLO")

if __name__ == "__main__":
    main()