import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

st.set_page_config(layout='wide')
df = pd.read_csv('india.csv')

list_of_states = list(df['State'].unique())
list_of_states.insert(0,'Overall India')
st.sidebar.title('India Data Visualisation')

selected_state = st.sidebar.selectbox('Select a state',list_of_states)
primary = st.sidebar.selectbox('Select Primary Parameter',
                               sorted(df.columns[5:])) # Représentatiopn des données par la taille
secondary = st.sidebar.selectbox('Select Secondary Parameter',
                                 sorted(df.columns[5:])) # Représentatiopn des données par la couleur
plot = st.sidebar.button('Plot Graph')

if plot:
    st.text('Size represents Primary Parameter')
    st.text('Color represents Secondary Parameter')
    if selected_state == 'Overall India':
        # Plot for India
        fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude',
        size=primary, color=secondary, zoom=4, size_max=35, mapbox_style='open-street-map',
        width=1200, height=700, hover_name='District')
        st.plotly_chart(fig,use_container_width=True)

        # Plot for States
        state_df = df[df.State == selected_state]
        fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude',
        size=primary, color=secondary, zoom=4, size_max=35, mapbox_style='open-street-map',
        width=1200, height=700, hover_name='District')