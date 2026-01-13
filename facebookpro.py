import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock forecast App")
stocks = ('GOOG','AAPL','MSFT','GME')
selected_stock = st.selectbox('Select a dataset for prediction' ,stocks)
n_months = st.slider('Months of predictions:',1,5)
periods = n_months*30

@st.cache_data
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('loading data...')
data = load_data(selected_stock)
data_load_state.text('loading data... done!')
st.subheader('Raw data')
data.columns = data.columns.droplevel(1)
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig= go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name='stock_open'))
    fig.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name='stock_close'))
    fig.layout.update(title_text='Time series with range slider',
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

# Variable name change
df = data[["Date","Close"]]
df = df.rename(columns={'Date':'ds','Close':'y'})

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=periods)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast Data')
st.write(forecast.tail())
st.write(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)
