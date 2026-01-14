import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle

@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

st.sidebar.image('VictoryRoad.png')

def main():
    st.markdown("<h2 style='text-align:center; color:brown;'> Streamlit Diabetis App </h1>",
                 unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:brown;'> Diabetis study in Cameroon </h2>",
                 unsafe_allow_html=True)
    menu = ['Home','Analysis','Data Visualisation','Machine Learning']
    choice = st.sidebar.selectbox('Select Menu',menu)
    data = load_data('diabetes.csv')
    if choice == 'Home':
        left,middle,right = st.columns((2,3,2))
        with middle:
            st.image('VictoryRoad.png')
        st.write("This is an app that will analyse diabetes Datas with some python tools")
        st.subheader("Diabetis Informations")
        st.write("In Cameroon, the prevalence of diabetis in adults in urban areas is")

    elif choice == 'Analysis':
        st.subheader('Diabetis Dataset')
        st.write(data.head())

        if st.checkbox('Summary'):
            st.write(data.describe())
        elif st.checkbox('Correlation'):
            fig = plt.figure(figsize=(15,15))
            st.write(sns.heatmap(data.corr(),annot=True))
            st.pyplot(fig)

    elif choice == 'Data Visualisation':
        if st.checkbox('Countplot'):
            fig1 = plt.figure(figsize=(13,5))
            sns.countplot(x='Age',data=data)
            st.pyplot(fig1)

        elif st.checkbox('Scatterplot'):
            fig2 = plt.figure(figsize=(8,8))
            sns.scatterplot(x='Glucose',y='Age',data=data,hue='Outcome')
            st.pyplot(fig2)

    elif choice == 'Machine Learning':
        tab1,tab2,tab3 = st.tabs([":clipboard: Data",":bar_chart: Visualisation",":mask: :smile: Prediction"])
        uploaded_file = st.sidebar.file_uploader('Upload your Input CSV File', type=["csv"])
        if uploaded_file:
            df = load_data(uploaded_file)

            with tab1:
                st.subheader('Loaded dataset')
                st.write(df)
            
            with tab2:
                st.subheader('Histogram Glucose')
                fig = plt.figure(figsize=(8,8))
                sns.histplot(x='Glucose',data=df)
                st.pyplot(fig)

            with tab3:
                model = pickle.load(open('model_dump.pkl','rb'))
                prediction = model.predict(df)
                st.subheader('Prediction')
                pp = pd.DataFrame(prediction,columns=['Prediction'])
                ndf = pd.concat([df,pp],axis=1)
                ndf.Prediction.replace(0,'No Diabete',inplace=True)
                ndf.Prediction.replace(1,'Diabete',inplace=True)
                st.write(ndf)


if __name__=='__main__':
    main()