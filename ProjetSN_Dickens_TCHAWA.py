import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =============================
# PALETTE DE COULEURS
# =============================
COLORS = {
    # Couleurs principales
    "primary_green": "#2d5016",
    "secondary_green": "#4a7c59",
    "light_green": "#90be6d",
    "background": "#f1faee",
    
    # Couleurs maladies
    "rust": "#e76f51",
    "blight": "#d62828",
    "mildew": "#f4a261",
    
    # Couleurs système
    "warning": "#ffd166",
    "success": "#06a77d",
    "text_dark": "#1a1a1a",
    "text_light": "#ffffff",
}

# =============================
# IMAGES DES MALADIES (EN LIGNE)
# =============================
DISEASE_IMAGES = {
    "healthy": "https://i.pinimg.com/1200x/1b/c4/2c/1bc42c7d47f8cd74674214eb1a929781.jpg",
    "rust": "https://www.planetnatural.com/wp-content/uploads/2012/12/common-rust-disease-920x518.webp",
    "mildew": "https://i.pinimg.com/1200x/28/a5/17/28a517647f0609040e73517dfbacf088.jpg",
    "blight": "https://i.pinimg.com/1200x/73/e1/bb/73e1bb26d91c4833d9cdbbdb1440898a.jpg",
    "logo":"https://i.pinimg.com/1200x/3f/6b/40/3f6b40950e452d3fd8263f718a6c31e6.jpg",
    "logos":"https://i.pinimg.com/736x/93/1d/02/931d0237e282905aa5c59b691471d08f.jpg"
}


# =============================
# CONFIGURATION PAGE
# =============================
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CSS PERSONNALISÉ AMÉLIORÉ
# =============================
st.markdown(
    f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    /* Arrière-plan général */
    .stApp {{
        background: linear-gradient(135deg, {COLORS['background']} 0%, #e8f5e9 100%);
        font-family: 'Poppins', sans-serif;
    }}
    
    /* Titres */
    h1 {{
        color: {COLORS['primary_green']};
        font-weight: 700;
        border-bottom: 4px solid {COLORS['light_green']};
        padding-bottom: 15px;
        margin-bottom: 25px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    
    h2 {{
        color: {COLORS['secondary_green']};
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 20px;
    }}
    
    h3 {{
        color: {COLORS['secondary_green']};
        font-weight: 500;
    }}
    
    /* Boutons */
    .stButton>button {{
        background: linear-gradient(135deg, {COLORS['primary_green']} 0%, {COLORS['secondary_green']} 100%);
        color: {COLORS['text_light']};
        border-radius: 12px;
        font-weight: 600;
        border: none;
        padding: 14px 28px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(45, 80, 22, 0.3);
        letter-spacing: 0.5px;
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, {COLORS['secondary_green']} 0%, {COLORS['light_green']} 100%);
        box-shadow: 0 6px 20px rgba(45, 80, 22, 0.4);
        transform: translateY(-2px);
    }}
    
    .stButton>button:active {{
        transform: translateY(0px);
        box-shadow: 0 3px 10px rgba(45, 80, 22, 0.3);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['primary_green']} 0%, {COLORS['secondary_green']} 100%);
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }}
    
    [data-testid="stSidebar"] * {{
        color: {COLORS['text_light']} !important;
    }}
    
    [data-testid="stSidebar"] .stRadio > label {{
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stSidebar"] .stRadio > label:hover {{
        background-color: rgba(255, 255, 255, 0.2);
    }}
    
    /* Messages d'info/succès/warning/error */
    .stSuccess {{
        background: linear-gradient(135deg, {COLORS['success']} 0%, #52b788 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #2d5016;
        box-shadow: 0 4px 15px rgba(6, 167, 125, 0.3);
    }}
    
    .stWarning {{
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid {COLORS['rust']};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);
    }}
    
    .stError {{
        background: linear-gradient(135deg, {COLORS['blight']} 0%, #c82333 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #721c24;
        box-shadow: 0 4px 15px rgba(214, 40, 40, 0.3);
    }}
    
    .stInfo {{
        background: linear-gradient(135deg, #d1f2eb 0%, #a8e6cf 100%);
        border-left: 5px solid {COLORS['primary_green']};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(45, 80, 22, 0.2);
    }}
    
    /* Cartes métriques */
    [data-testid="stMetricValue"] {{
        color: {COLORS['primary_green']};
        font-weight: 700;
        font-size: 2em;
    }}
    
    [data-testid="stMetric"] {{
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid {COLORS['light_green']};
    }}
    
    /* Formulaires */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {{
        border-radius: 10px;
        border: 2px solid {COLORS['light_green']};
        padding: 10px;
        font-size: 15px;
        transition: all 0.3s ease;
    }}
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus,
    .stNumberInput>div>div>input:focus {{
        border-color: {COLORS['primary_green']};
        box-shadow: 0 0 0 3px rgba(45, 80, 22, 0.1);
    }}
    
    /* Badges de maladie */
    .disease-badge {{
        display: inline-block;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: 700;
        margin: 10px;
        font-size: 18px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }}
    
    .disease-badge:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }}
    
    .badge-rust {{
        background: linear-gradient(135deg, {COLORS['rust']} 0%, #d94829 100%);
        color: {COLORS['text_light']};
    }}
    
    .badge-blight {{
        background: linear-gradient(135deg, {COLORS['blight']} 0%, #a91d1d 100%);
        color: {COLORS['text_light']};
    }}
    
    .badge-mildew {{
        background: linear-gradient(135deg, {COLORS['mildew']} 0%, #e09f3e 100%);
        color: {COLORS['text_dark']};
    }}
    
    /* Image containers */
    .disease-image-container {{
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin: 20px 0;
        transition: all 0.3s ease;
    }}
    
    .disease-image-container:hover {{
        transform: scale(1.02);
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    }}
    
    /* DataFrames */
    .dataframe {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    /* Cards personnalisées */
    .custom-card {{
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid {COLORS['light_green']};
        transition: all 0.3s ease;
    }}
    
    .custom-card:hover {{
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        transform: translateY(-3px);
    }}
    
    /* Séparateur stylisé */
    hr {{
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, {COLORS['light_green']}, transparent);
        margin: 30px 0;
    }}
    
    /* Animation de chargement */
    @keyframes pulse {{
        0%, 100% {{
            opacity: 1;
        }}
        50% {{
            opacity: 0.5;
        }}
    }}
    
    .loading {{
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }}
    
    /* Animation fade-in */
    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.8s ease-out forwards;
    }}
    
    .fade-in-delay-1 {{
        animation: fadeIn 0.8s ease-out 0.2s forwards;
        opacity: 0;
    }}
    
    .fade-in-delay-2 {{
        animation: fadeIn 0.8s ease-out 0.4s forwards;
        opacity: 0;
    }}
    
    .fade-in-delay-3 {{
        animation: fadeIn 0.8s ease-out 0.6s forwards;
        opacity: 0;
    }}
    
    .fade-in-delay-4 {{
        animation: fadeIn 0.8s ease-out 0.8s forwards;
        opacity: 0;
    }}
    
    .fade-in-slow {{
        animation: fadeIn 1.2s ease-out forwards;
    }}
    
    /* Animation pour les images */
    @keyframes scaleIn {{
        from {{
            opacity: 0;
            transform: scale(0.9);
        }}
        to {{
            opacity: 1;
            transform: scale(1);
        }}
    }}
    
    .scale-in {{
        animation: scaleIn 1s ease-out forwards;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# CHARGEMENT MODELE & ENCODEURS
# =============================
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load("decision_tree_model.pkl")
        encoders = joblib.load("encoders.pkl")
        return model, encoders
    except FileNotFoundError:
        st.error("⚠️ Erreur : Les fichiers 'decision_tree_model.pkl' ou 'encoders.pkl' sont introuvables.")
        st.stop()

model, encoders = load_model_and_encoders()

# =============================
# CHARGEMENT DATASET
# =============================
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("plant_disease_dataset.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ Erreur : Le fichier 'plant_disease_dataset.csv' est introuvable.")
        st.stop()

df = load_dataset()
feature_columns = df.columns[:-1]
target_column = df.columns[-1]

# Initialiser l'encodeur cible
target_label_encoder = LabelEncoder()
target_label_encoder.fit(df[target_column].astype(str))

# =============================
# SIDEBAR
# =============================
st.sidebar.markdown("## 🌱 Navigation")
menu = st.sidebar.radio(
    "",
    ["🏠 Accueil", "🔍 Prédiction", "📊 Analyse", "📁 Prédiction CSV"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
    <h4 style='margin-top: 0;'>📌 Informations</h4>
    <p style='font-size: 14px; margin: 5px 0;'>✓ Decision Tree optimisé</p>
    <p style='font-size: 14px; margin: 5px 0;'>✓ Encodage cohérent</p>
    <p style='font-size: 14px; margin: 5px 0;'>✓ Prédictions fiables</p>
    <p style='font-size: 14px; margin: 5px 0;'>✓ Visualisations interactives</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================
# PAGE ACCUEIL
# =============================
if menu == "🏠 Accueil":
    # Titre principal avec style et animation
    st.markdown(
        """
        <div class='fade-in' style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 3.5em; margin-bottom: 10px;'>🌿 Plant Disease Prediction</h1>
            <p style='font-size: 1.2em; color: #4a7c59;'>Système intelligent de détection des maladies des plantes</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Afficher l'image de plante saine avec animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="disease-image-container scale-in">', unsafe_allow_html=True)
        st.image(DISEASE_IMAGES["healthy"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section avec cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div class='custom-card fade-in-delay-1'>
                <h3>🎯 Objectif de l'application</h3>
                <p>Cette application permet de <strong>prédire la maladie d'une plante</strong>
                à partir de ses caractéristiques physiques et environnementales en utilisant 
                un <strong>modèle Decision Tree optimisé</strong>.</p>
                <p>Notre système analyse plusieurs paramètres comme la taille des feuilles, 
                le diamètre de la tige, le type de sol, les conditions météorologiques 
                et l'utilisation de pesticides pour fournir un diagnostic précis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div class='custom-card fade-in-delay-2'>
                <h3>🧠 Technologies utilisées</h3>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>🐍 <strong>Python</strong> - Langage de programmation</li>
                    <li>🤖 <strong>Scikit-learn</strong> - Machine Learning</li>
                    <li>🎨 <strong>Streamlit</strong> - Interface web</li>
                    <li>💾 <strong>Joblib</strong> - Sérialisation de modèles</li>
                    <li>📊 <strong>Plotly</strong> - Visualisations interactives</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class='custom-card fade-in-delay-4'>
                <h3>🚀 Comment démarrer ?</h3>
                <p>Utilisez le menu de navigation à gauche pour :</p>
                <ol>
                    <li><strong>🔍 Prédiction</strong> - Analyser une plante individuelle</li>
                    <li><strong>📊 Analyse</strong> - Explorer les données et le modèle</li>
                    <li><strong>📁 Prédiction CSV</strong> - Traiter plusieurs plantes en lot</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="disease-image-container scale-in">', unsafe_allow_html=True)
        st.image(DISEASE_IMAGES["logo"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# Nouvelle section : Galerie des maladies
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        <div class='fade-in-delay-4' style='text-align: center;'>
            <h2>🖼️ Galerie des Maladies</h2>
            <p style='color: #4a7c59; font-size: 16px;'>Visualisez les symptômes de chaque maladie</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Afficher les 3 images côte à côte
    img_col1, img_col2, img_col3 = st.columns(3)
    
    with img_col1:
        st.markdown(
            f"""
            <div class='disease-image-container fade-in-delay-4' style='text-align: center;'>
                <div style='background: {COLORS['rust']}; color: white; padding: 10px; 
                            border-radius: 10px 10px 0 0; font-weight: bold;'>
                    🔴 RUST
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(DISEASE_IMAGES["rust"], use_container_width=True)

        st.markdown(
            """
            <div style='background: #fff; padding: 10px; border-radius: 0 0 10px 10px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <small>Pustules orangées</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with img_col2:
        st.markdown(
            f"""
            <div class='disease-image-container fade-in-delay-4' style='text-align: center;'>
                <div style='background: {COLORS['blight']}; color: white; padding: 10px; 
                            border-radius: 10px 10px 0 0; font-weight: bold;'>
                    🚨 BLIGHT
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(DISEASE_IMAGES["blight"], use_container_width=True)

        st.markdown(
            """
            <div style='background: #fff; padding: 10px; border-radius: 0 0 10px 10px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <small>Taches noires nécrotiques</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with img_col3:
        st.markdown(
            f"""
            <div class='disease-image-container fade-in-delay-4' style='text-align: center;'>
                <div style='background: {COLORS['mildew']}; color: #1a1a1a; padding: 10px; 
                            border-radius: 10px 10px 0 0; font-weight: bold;'>
                    ⚠️ MILDEW
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(DISEASE_IMAGES["mildew"], use_container_width=True)

        st.markdown(
            """
            <div style='background: #fff; padding: 10px; border-radius: 0 0 10px 10px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <small>Duvet blanc poudreux</small>
            </div>
            """,
            unsafe_allow_html=True
        )

# =============================
# PAGE PREDICTION SIMPLE
# =============================
elif menu == "🔍 Prédiction":
    st.header("🔍 Prédiction individuelle")

    st.write("Veuillez entrer les caractéristiques de la plante pour obtenir une prédiction.")

    with st.form("prediction_form"):
        user_input = {}

        for col in feature_columns:
            if df[col].dtype == "object":
                user_input[col] = st.selectbox(
                    f"**{col.replace('_', ' ').title()}**",
                    encoders[col].classes_
                )
            else:
                label = f"**{col.replace('_', ' ').title()}**"
                if col in ['leaf_length', 'leaf_width', 'stem_diameter']:
                    label += " (cm)"
                user_input[col] = st.number_input(
                    label,
                    min_value=float(df[col].min()),
                    max_value=float(df[col].max()),
                    value=float(df[col].mean())
                )

        submitted = st.form_submit_button("🚀 Prédire")

    if submitted:
        input_df = pd.DataFrame([user_input])

        for col in feature_columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        # S'assurer que les colonnes sont dans le bon ordre
        input_df = input_df[feature_columns]

        prediction_index = model.predict(input_df)[0]
        predicted_label = target_label_encoder.inverse_transform([prediction_index])[0]

        st.balloons()
        st.success(f"🌱 Maladie prédite : **{predicted_label.upper()}**")

        # =====================================================================
        # CONSEILS SPÉCIFIQUES PAR MALADIE
        # =====================================================================
        st.markdown("---")
        st.subheader("💡 Conseils et informations spécifiques")

        if predicted_label.lower() == "rust":
            st.warning(
                "**⚠️ ROUILLE (RUST) DÉTECTÉE**\n\n"
                "**Symptômes caractéristiques :**\n"
                "• Pustules poudreuses orangées, brunes ou rougeâtres sur les feuilles\n"
                "• Apparition principalement sur la face inférieure des feuilles\n"
                "• Jaunissement et chute prématurée du feuillage\n"
                "• Affaiblissement général de la plante\n\n"
                "**Actions recommandées :**\n"
                "1. **Suppression immédiate** : Retirez et détruisez toutes les feuilles infectées (ne pas composter)\n"
                "2. **Amélioration de la circulation d'air** : Taillez pour espacer le feuillage et réduire l'humidité\n"
                "3. **Arrosage adapté** : Arrosez uniquement au pied de la plante, évitez de mouiller le feuillage\n"
                "4. **Traitement fongicide** : Appliquez un fongicide à base de soufre ou de cuivre selon les recommandations\n"
                "5. **Prévention** : Maintenez un espacement adéquat entre les plants et assurez une bonne aération\n\n"
                "⏰ **Agissez rapidement** : La rouille se propage facilement par les spores dans des conditions humides."
            )
        
        elif predicted_label.lower() == "blight":
            st.error(
                "**🚨 MILDIOU (BLIGHT) DÉTECTÉ**\n\n"
                "**Symptômes caractéristiques :**\n"
                "• Taches brunes ou noires sur les feuilles, souvent avec un halo jaune\n"
                "• Lésions humides qui s'étendent rapidement\n"
                "• Nécrose des tiges et brunissement des tissus\n"
                "• Pourriture potentielle des fruits/légumes\n"
                "• Dégâts rapides en conditions humides\n\n"
                "**Actions urgentes :**\n"
                "1. **Isolation** : Éloignez immédiatement les plants infectés des plants sains\n"
                "2. **Élimination drastique** : Coupez et brûlez les parties infectées (pas de compostage !)\n"
                "3. **Désinfection des outils** : Nettoyez tous les outils de taille à l'alcool entre chaque coupe\n"
                "4. **Traitement fongicide préventif** : Appliquez un fongicide systémique sur les plants sains environnants\n"
                "5. **Gestion de l'humidité** : Réduisez drastiquement l'arrosage et assurez un drainage optimal\n"
                "6. **Rotation des cultures** : Ne replantez pas la même espèce au même endroit pendant 2-3 ans\n\n"
                "⚠️ **ATTENTION** : Le mildiou est extrêmement contagieux et peut détruire une culture entière en quelques jours."
            )
        
        elif predicted_label.lower() == "mildew":
            st.warning(
                "**⚠️ OÏDIUM (MILDEW) DÉTECTÉ**\n\n"
                "**Symptômes caractéristiques :**\n"
                "• Duvet blanc poudreux ou grisâtre sur les feuilles\n"
                "• Déformation et recroquevillement des jeunes feuilles\n"
                "• Décoloration jaunâtre sous le duvet blanc\n"
                "• Affaiblissement de la photosynthèse\n"
                "• Développement favorisé par temps chaud et sec\n\n"
                "**Actions recommandées :**\n"
                "1. **Traitement naturel initial** : Pulvérisez une solution de bicarbonate de soude (1 c. à soupe/litre d'eau + quelques gouttes de savon)\n"
                "2. **Suppression sélective** : Retirez les parties les plus atteintes\n"
                "3. **Amélioration des conditions** : Augmentez la circulation d'air et réduisez l'humidité relative\n"
                "4. **Arrosage matinal** : Arrosez tôt le matin pour permettre au feuillage de sécher rapidement\n"
                "5. **Fongicide si nécessaire** : Utilisez un fongicide anti-oïdium en cas d'infestation sévère\n"
                "6. **Soufre en poudre** : Le soufre est efficace en prévention et traitement léger\n\n"
                "💡 **Bon à savoir** : L'oïdium est plus facile à traiter que le mildiou mais nécessite une action rapide."
            )
        
        else:
            st.info(
                f"**Maladie identifiée : {predicted_label}**\n\n"
                "Pour des conseils de traitement détaillés sur cette maladie spécifique, "
                "consultez un expert en phytopathologie ou un service agricole local."
            )
            
# =============================
# PAGE ANALYSE AVEC GRAPHIQUES AMÉLIORÉS
# =============================
elif menu == "📊 Analyse":
    st.markdown("<h1>📊 Analyse du Modèle et des Données</h1>", unsafe_allow_html=True)

    # Section 1: Importance des caractéristiques
    st.markdown("## 🔑 Importance des Caractéristiques")
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        # Créer un graphique Plotly avec style amélioré
        fig_importance = go.Figure()
        
        fig_importance.add_trace(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(
                color=importance_df['Importance'],
                colorscale=[[0, COLORS['light_green']], 
                           [0.5, COLORS['secondary_green']], 
                           [1, COLORS['primary_green']]],
                line=dict(color='white', width=2)
            ),
            text=importance_df['Importance'].round(3),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig_importance.update_layout(
            title={
                'text': 'Importance des caractéristiques dans le modèle',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': COLORS['primary_green']}
            },
            xaxis_title="Score d'importance",
            yaxis_title="Caractéristiques",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins', size=12),
            margin=dict(l=20, r=20, t=80, b=20),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(144, 190, 109, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(144, 190, 109, 0.3)'
            ),
            yaxis=dict(
                showgrid=False
            )
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown(
            """
            <div class='custom-card'>
                <p><strong>💡 Interprétation :</strong> Les caractéristiques avec les scores les plus élevés 
                ont le plus d'influence sur les prédictions du modèle. Elles sont les plus déterminantes 
                pour identifier la maladie de la plante.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Le modèle ne fournit pas d'informations sur l'importance des caractéristiques.")

    st.markdown("---")

    # Section 2: Répartition des maladies
    st.markdown("## 📈 Répartition des Maladies dans le Dataset")
    
    disease_counts = df[target_column].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Graphique en barres avec Plotly
        fig_diseases = go.Figure()
        
        colors_map = {
            'rust': COLORS['rust'],
            'blight': COLORS['blight'],
            'mildew': COLORS['mildew']
        }
        
        bar_colors = [colors_map.get(disease, COLORS['primary_green']) for disease in disease_counts.index]
        
        fig_diseases.add_trace(go.Bar(
            x=disease_counts.index,
            y=disease_counts.values,
            marker=dict(
                color=bar_colors,
                line=dict(color='white', width=3),
                cornerradius=15
            ),
            text=disease_counts.values,
            textposition='outside',
            textfont=dict(size=16, color=COLORS['text_dark'], family='Poppins', weight='bold'),
            hovertemplate='<b>%{x}</b><br>Nombre: %{y}<br>Pourcentage: %{customdata:.1f}%<extra></extra>',
            customdata=[100 * count / disease_counts.sum() for count in disease_counts.values]
        ))
        
        fig_diseases.update_layout(
            title={
                'text': 'Distribution des maladies',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 22, 'color': COLORS['primary_green'], 'family': 'Poppins'}
            },
            xaxis_title="Type de maladie",
            yaxis_title="Nombre de cas",
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins', size=13),
            margin=dict(l=20, r=20, t=80, b=20),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=14, color=COLORS['text_dark'])
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(144, 190, 109, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(144, 190, 109, 0.3)'
            )
        )
        
        st.plotly_chart(fig_diseases, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Statistiques")
        
        # Métriques avec style
        for disease in ['rust', 'blight', 'mildew']:
            count = disease_counts.get(disease, 0)
            percentage = (count / disease_counts.sum() * 100)
            
            if disease == 'rust':
                emoji = "🦠"
                label = "Rust"
            elif disease == 'blight':
                emoji = "🚨"
                label = "Blight"
            else:
                emoji = "⚠️"
                label = "Mildew"
            
            st.metric(
                label=f"{emoji} {label}",
                value=f"{count}",
                delta=f"{percentage:.1f}%"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            label="📦 Total",
            value=f"{disease_counts.sum()}"
        )
    
    st.markdown("---")
    
    # Section 3: Graphique circulaire (Pie chart)
    st.markdown("## 🥧 Proportion des Maladies")
    
    fig_pie = go.Figure()
    
    fig_pie.add_trace(go.Pie(
        labels=disease_counts.index,
        values=disease_counts.values,
        hole=0.4,
        marker=dict(
            colors=[colors_map.get(disease, COLORS['primary_green']) for disease in disease_counts.index],
            line=dict(color='white', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(size=16, color='white', family='Poppins', weight='bold'),
        hovertemplate='<b>%{label}</b><br>Nombre: %{value}<br>Pourcentage: %{percent}<extra></extra>'
    ))
    
    fig_pie.update_layout(
        title={
            'text': 'Répartition proportionnelle des maladies',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': COLORS['primary_green'], 'family': 'Poppins'}
        },
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Poppins', size=13),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=14)
        )
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

# =============================
# PAGE PREDICTION CSV
# =============================
elif menu == "📁 Prédiction CSV":
    st.markdown("<h1>📁 Prédiction par Fichier CSV</h1>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='custom-card'>
            <h3>📋 Instructions</h3>
            <p>Téléchargez un fichier CSV contenant les mêmes colonnes de caractéristiques 
            que le dataset d'entraînement pour obtenir des prédictions en masse.</p>
            <p><strong>Colonnes requises :</strong> {', '.join(feature_columns)}</p>
            <p style='color: {COLORS['rust']};'><strong>Important :</strong> Le fichier ne doit PAS contenir 
            la colonne de maladie (disease_type).</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📤 Importer un fichier CSV",
        type=["csv"],
        help="Le fichier doit contenir les colonnes de caractéristiques attendues"
    )

    if uploaded_file is not None:
        csv_data = pd.read_csv(uploaded_file)
        
        st.markdown("### 📄 Aperçu du fichier importé")
        st.dataframe(csv_data.head(10), use_container_width=True)

        missing_cols = [col for col in feature_columns if col not in csv_data.columns]
        if missing_cols:
            st.error(
                f"""
                ❌ **Erreur de format**
                
                Les colonnes suivantes sont manquantes dans votre fichier CSV :
                **{', '.join(missing_cols)}**
                
                Veuillez vous assurer que votre fichier contient toutes les colonnes requises.
                """
            )
        else:
            if st.button("📊 Lancer les Prédictions", use_container_width=True):
                with st.spinner('🔄 Prédictions en cours...'):
                    # Appliquer les encodeurs
                    for col in feature_columns:
                        if col in encoders:
                            try:
                                csv_data[col] = encoders[col].transform(csv_data[col])
                            except ValueError as e:
                                st.error(
                                    f"""
                                    ❌ **Erreur d'encodage pour la colonne '{col}'**
                                    
                                    {str(e)}
                                    
                                    Assurez-vous que toutes les valeurs sont valides.
                                    """
                                )
                                st.stop()

                    # Effectuer les prédictions
                    predictions_indices = model.predict(csv_data[feature_columns])
                    predicted_labels_csv = target_label_encoder.inverse_transform(predictions_indices)
                    csv_data["Predicted_Disease"] = predicted_labels_csv

                st.success("✅ **Prédictions terminées avec succès !**")
                
                # Statistiques des prédictions
                st.markdown("---")
                st.markdown("## 📊 Résumé des Prédictions")
                
                prediction_counts = pd.Series(predicted_labels_csv).value_counts()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🦠 Rust", prediction_counts.get('rust', 0))
                with col2:
                    st.metric("🚨 Blight", prediction_counts.get('blight', 0))
                with col3:
                    st.metric("⚠️ Mildew", prediction_counts.get('mildew', 0))
                with col4:
                    st.metric("📦 Total", len(predicted_labels_csv))
                
                # Graphique des prédictions
                st.markdown("### 📈 Visualisation des résultats")
                
                colors_map = {
                    'rust': COLORS['rust'],
                    'blight': COLORS['blight'],
                    'mildew': COLORS['mildew']
                }
                
                bar_colors = [colors_map.get(disease, COLORS['primary_green']) for disease in prediction_counts.index]
                
                fig_predictions = go.Figure()
                
                fig_predictions.add_trace(go.Bar(
                    x=prediction_counts.index,
                    y=prediction_counts.values,
                    marker=dict(
                        color=bar_colors,
                        line=dict(color='white', width=3),
                        cornerradius=15
                    ),
                    text=prediction_counts.values,
                    textposition='outside',
                    textfont=dict(size=16, family='Poppins', weight='bold'),
                    hovertemplate='<b>%{x}</b><br>Nombre: %{y}<extra></extra>'
                ))
                
                fig_predictions.update_layout(
                    title={
                        'text': 'Distribution des prédictions',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'color': COLORS['primary_green']}
                    },
                    xaxis_title="Maladie prédite",
                    yaxis_title="Nombre de cas",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Poppins'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(144, 190, 109, 0.2)'
                    )
                )
                
                st.plotly_chart(fig_predictions, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📋 Aperçu des résultats")
                st.dataframe(csv_data.head(20), use_container_width=True)

                # Bouton de téléchargement
                csv_download = csv_data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Télécharger les Résultats Complets",
                    csv_download,
                    "predictions_plant_disease.csv",
                    "text/csv",
                    key="download-csv",
                    use_container_width=True
                )

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; padding: 20px; color: {COLORS['secondary_green']};'>
        <p style='margin: 0; font-size: 14px;'>
            🎓 <strong>Projet Machine Learning</strong> – Decision Tree – Prédiction de maladies des plantes
        </p>
        <p style='margin: 5px 0; font-size: 13px;'>
            🦠 Rust | 🚨 Blight | ⚠️ Mildew
        </p>
        <p style='margin: 5px 0; font-size: 13px; color: {COLORS['light_green']}'>
            Développé par <strong>Dickens Tchawa</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)