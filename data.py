import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title('File Uploader')
st.subheader('Input Csv')
uploaded_file = st.file_uploader('Choose a file')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader('Dataframe')
    st.write(df)
    col1,col2 = st.columns(2)
    with col1:
        fig1 = plt.figure(figsize=(16,16))
        sns.scatterplot(x='EstimatedSalary', y='Age', hue='Purchased', data=df)
        st.pyplot(fig1)
    with col2:
        fig2 = plt.figure(figsize=(14,14))
        sns.histplot(df.Age)
        st.pyplot(fig2)