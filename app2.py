import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import time

# Chargement du modele
with open('reg.pkl','rb') as file:
    model = pickle.load(file)

st.title('Predicteur de charges d assurances medicales')

# Ajout d'une animation
with st.spinner('Chargement du Modele....'):
    time.sleep(1)

# Entree des inputs
col1,col2 = st.columns(2)
with col1:
    age = st.slider('Age',10,100,24)
with col2:
    sex = st.selectbox('Sexe',['male','female'])

col3,col4 = st.columns(2)
with col3:
    bmi = st.number_input('BMI (Indice Masse Corporelle)',10,50,25)
with col4:
    children = st.slider('Nombre d enfants',0,5,1)

col5,col6 = st.columns(2)
with col5:
    smoker = st.selectbox('Fumeur?',['yes','no'])
with col6:
    region = st.selectbox('Region',['southwest','southeast','northwest','northeast'])

# ENCODAGE
sex_encoded = 1 if sex == 'male' else 0
smoker_encoded = 1 if smoker == 'yes' else 0
region_dict = {"southwest": 0.24308153, "southeast":0.27225131, "northwest":0.24233358, "northeast":0.27225131}
region_encoded = region_dict[region]

# PREPARATION DES DONNEES
input_data = [[age,sex_encoded,bmi,children,smoker_encoded,region_encoded]]

# PREDICTION
if st.button('Predire les Charges'):
    with st.spinner('Calcul en cours...'):
        prediction = model.predict(input_data)[0]
        time.sleep(1)
    st.success("Prediction Terminee")
    st.markdown(f"<h2 style='text-align:center; color:brown;'>Charges Medicales estimees: **${round(prediction)}**</h2>",unsafe_allow_html=True)
    st.balloons()



