import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import base64
import yfinance as yf
import urllib

st.title('S&P 500 App')
st.markdown("""This app retrieves the S&P Table from wikipeida and its corresponding prizes to yahoo finance""")
st.markdown("**Data Source** [wikipedia] : https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

st.sidebar.header('User Input Features')

# Scrapping du tableau wikipedia
@st.cache_data
def load_data():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Add a user agent to avoid being blocked by wikipedia
    headers={"User-Agent":"Mozilla/5.0"}
    req=urllib.request.Request(url,headers=headers)
    
    with urllib.request.urlopen(req) as response :
        html = response.read()

    # Extract the first table (S&P)
    df = pd.read_html(html,header=0)[0]
    return df
df = load_data()
st.write(df)
sector=df.groupby('GICS Sector')

# Sidebar sector selection
sorted_sector_unique = sorted(df['GICS Sector'].unique())
selected_sector = st.sidebar.multiselect('Sector',sorted_sector_unique)

# Filter data
df_selected_sector = df[(df['GICS Sector'].isin(selected_sector))]
st.header('Display companies in selected sector')
st.write('Data Dimension: '+str(df_selected_sector.shape[0])+ 
    ' rows and '+str(df_selected_sector.shape[1])+ ' columns')

# Yahoo Finance


# Plot closing price
def price_plot (symbol):
    try:
        data = yf.download(
            tickers= list(df_selected_sector[:10].Symbol),
            period='ytd',
            interval='1d',
            group_by='ticker',
            auto_adjust=True)

        df = pd.DataFrame(data[symbol]['Close'])
        df['Date']=df.index
        plt.fill_between(df['Date'],df['Close'],color='skyblue')
        plt.plot(df['Date'],df['Close'],color='skyblue')
        plt.title(symbol,fontweight='bold')
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Closing Price', fontweight='bold')
        return st.pyplot(plt)
    except:
        print('No selected sector')

num_company = st.sidebar.slider('Number of companies', 1,5)
if st.button('Show plots'):
    st.header('Stock closing price')
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)