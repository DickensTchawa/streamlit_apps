import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# PALETTE DE COULEURS
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

# CONFIGURATION PAGE
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌱",
    layout="wide"
)

# THEME CSS
st.markdown(
    f"""
    <style>
    /* Arrière-plan général */
    .stApp {{
        background-color: {COLORS['background']};
    }}
    
    /* Titres */
    h1 {{
        color: {COLORS['primary_green']};
        font-weight: 700;
        border-bottom: 3px solid {COLORS['light_green']};
        padding-bottom: 10px;
    }}
    
    h2, h3 {{
        color: {COLORS['secondary_green']};
        font-weight: 600;
    }}
    
    /* Boutons */
    .stButton>button {{
        background: linear-gradient(135deg, {COLORS['primary_green']} 0%, {COLORS['secondary_green']} 100%);
        color: {COLORS['text_light']};
        border-radius: 10px;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, {COLORS['secondary_green']} 0%, {COLORS['light_green']} 100%);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['primary_green']} 0%, {COLORS['secondary_green']} 100%);
    }}
    
    [data-testid="stSidebar"] * {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* Messages de succès */
    .stSuccess {{
        background-color: {COLORS['success']};
        color: {COLORS['text_light']};
        border-radius: 8px;
        padding: 15px;
    }}
    
    /* Messages d'avertissement */
    .stWarning {{
        background-color: {COLORS['warning']};
        border-left: 5px solid {COLORS['rust']};
        border-radius: 8px;
        padding: 15px;
    }}
    
    /* Messages d'erreur */
    .stError {{
        background-color: {COLORS['blight']};
        color: {COLORS['text_light']};
        border-radius: 8px;
        padding: 15px;
    }}
    
    /* Cartes métriques */
    [data-testid="stMetricValue"] {{
        color: {COLORS['primary_green']};
        font-weight: 700;
    }}
    
    /* Formulaires */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {{
        border-radius: 8px;
        border: 2px solid {COLORS['light_green']};
    }}
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus,
    .stNumberInput>div>div>input:focus {{
        border-color: {COLORS['primary_green']};
        box-shadow: 0 0 0 2px {COLORS['light_green']}40;
    }}
    
    /* Badges de maladie */
    .disease-badge {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }}
    
    .badge-rust {{
        background-color: {COLORS['rust']};
        color: {COLORS['text_light']};
    }}
    
    .badge-blight {{
        background-color: {COLORS['blight']};
        color: {COLORS['text_light']};
    }}
    
    .badge-mildew {{
        background-color: {COLORS['mildew']};
        color: {COLORS['text_dark']};
    }}
    
    /* DataFrames */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
    }}
    
    /* Info boxes */
    .stInfo {{
        background-color: {COLORS['light_green']}30;
        border-left: 5px solid {COLORS['primary_green']};
        border-radius: 8px;
        padding: 15px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# CHARGEMENT MODELE & ENCODEURS
try:
    model = joblib.load("decision_tree_model.pkl")
    encoders = joblib.load("encoders.pkl")
except FileNotFoundError:
    st.error("Erreur : Les fichiers 'decision_tree_model.pkl' ou 'encoders.pkl' sont introuvables.")
    st.stop()

# CHARGEMENT DATASET & INITIALISATION ENCODEUR CIBLE
try:
    df = pd.read_csv("plant_disease_dataset.csv")
except FileNotFoundError:
    st.error("Erreur : Le fichier 'plant_disease_dataset.csv' est introuvable.")
    st.stop()

feature_columns = df.columns[:-1]
target_column = df.columns[-1]

# Initialiser et ajuster un LabelEncoder pour la colonne cible
target_label_encoder = LabelEncoder()
target_label_encoder.fit(df[target_column].astype(str))

# SIDEBAR
st.sidebar.title("🌱 Navigation")
menu = st.sidebar.radio(
    "Aller vers :",
    ["🏠 Accueil", "🔍 Prédiction", "📊 Analyse", "📁 Prédiction CSV"]
)

# PAGE ACCUEIL
if menu == "🏠 Accueil":
    st.markdown(
        "<h1 style='text-align:center;'>🌿 Prédiction de maladies des plantes</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### 🎯 Objectif de l'application
        Cette application permet de **prédire la maladie d'une plante**
        à partir de ses caractéristiques en utilisant un **Decision Tree optimisé**.

        ### 🦠 Maladies détectées
        - **Rust (Rouille)** : Maladie fongique causant des pustules orangées
        - **Blight (Mildiou)** : Maladie dévastatrice affectant feuilles et tiges
        - **Mildew (Oïdium)** : Champignon formant un duvet blanc sur les feuilles

        ### 🧠 Technologies/bibliothèques utilisées
        - Pandas
        - Scikit-learn
        - Streamlit
        - Joblib

        👉 Utilisez le menu à gauche pour commencer.
        """
    )

# PAGE PREDICTION SIMPLE
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

        prediction_index = model.predict(input_df)[0]
        predicted_label = target_label_encoder.inverse_transform([prediction_index])[0]

        st.balloons()
        st.success(f"🌱 Maladie prédite : **{predicted_label.upper()}**")

        # CONSEILS SPÉCIFIQUES PAR MALADIE (RUST,BLIGHT ou MILDEW)
        st.markdown("---")
        st.subheader("💡 Conseils et informations spécifiques")

        if predicted_label.lower() == "rust":
            st.markdown(
                f'<div class="disease-badge badge-rust">🦠 RUST - ROUILLE</div>',
                unsafe_allow_html=True
            )
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
            st.markdown(
                f'<div class="disease-badge badge-blight">🚨 BLIGHT - MILDIOU</div>',
                unsafe_allow_html=True
            )
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

            st.markdown(
                f'<div class="disease-badge badge-mildew">⚠️ MILDEW - OÏDIUM</div>',
                unsafe_allow_html=True
            )
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

# PAGE ANALYSE

elif menu == "📊 Analyse":
    st.header("📊 Analyse du modèle")

    st.subheader("🔑 Importance des caractéristiques")
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.warning("Le modèle chargé ne fournit pas d'informations sur l'importance des caractéristiques.")

    st.subheader("📈 Répartition des maladies dans le dataset")
    disease_counts = df[target_column].value_counts()
    st.bar_chart(disease_counts)
    
    # Affichage des statistiques
    st.markdown("---")
    st.subheader("📊 Statistiques des maladies")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🦠 Rust", disease_counts.get('rust', 0))
    with col2:
        st.metric("🚨 Blight", disease_counts.get('blight', 0))
    with col3:
        st.metric("⚠️ Mildew", disease_counts.get('mildew', 0))

# PAGE PREDICTION CSV
elif menu == "📁 Prédiction CSV":
    st.header("📁 Prédiction à partir d'un fichier CSV")

    st.write(
        "Téléchargez un fichier CSV contenant les mêmes colonnes de caractéristiques "
        "que votre dataset d'entraînement pour obtenir des prédictions en masse."
    )
    
    st.info(
        "**Colonnes requises :** " + ", ".join(feature_columns) + "\n\n"
        "Le fichier ne doit **pas** contenir la colonne de maladie (disease_type)."
    )

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        csv_data = pd.read_csv(uploaded_file)
        st.write("📄 Aperçu du fichier importé :", csv_data.head())

        missing_cols = [col for col in feature_columns if col not in csv_data.columns]
        if missing_cols:
            st.error(f"❌ Erreur : Les colonnes suivantes sont manquantes : **{', '.join(missing_cols)}**")
        else:
            if st.button("📊 Lancer les prédictions"):
                # Appliquer les encodeurs de caractéristiques
                for col in feature_columns:
                    if col in encoders:
                        try:
                            csv_data[col] = encoders[col].transform(csv_data[col])
                        except ValueError as e:
                            st.error(
                                f"❌ Erreur d'encodage pour la colonne **'{col}'** : {e}\n\n"
                                "Assurez-vous que toutes les valeurs sont valides."
                            )
                            st.stop()

                # Effectuer les prédictions
                predictions_indices = model.predict(csv_data[feature_columns])
                predicted_labels_csv = target_label_encoder.inverse_transform(predictions_indices)
                csv_data["Predicted_Disease"] = predicted_labels_csv

                st.success("✅ Prédictions terminées avec succès !")
                
                # Statistiques des prédictions
                st.markdown("---")
                st.subheader("📊 Résumé des prédictions")
                prediction_counts = pd.Series(predicted_labels_csv).value_counts()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🦠 Rust", prediction_counts.get('rust', 0))
                with col2:
                    st.metric("🚨 Blight", prediction_counts.get('blight', 0))
                with col3:
                    st.metric("⚠️ Mildew", prediction_counts.get('mildew', 0))
                
                st.markdown("---")
                st.write("📊 Aperçu des prédictions :")
                st.dataframe(csv_data.head(10))

                csv_download = csv_data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Télécharger les résultats complets",
                    csv_download,
                    "predictions_plant_disease.csv",
                    "text/csv",
                    key="download-csv"
                )

# FOOTER
st.markdown("---")
st.caption(
    "🎓 Projet Machine Learning – Decision Tree – Prédiction de maladies des plantes | "
    "🦠 Rust | 🚨 Blight | ⚠️ Mildew "
)
st.caption("TCHAWA Paul II Dickens, ING 5 ISI IL")