import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
from plotly import graph_objs as go
import yfinance

start = '2011-01-01'
today = date.today().strftime("%Y-%m-%d")

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, today)

st.subheader('Data from 2011-2022')
st.write(df.describe())

@st.cache
def load_data(user_input):
  st_data = yfinance.download(user_input, start, today)
  st_data.reset_index(inplace = True)
  return st_data

st_data = load_data(user_input)
st.subheader("Recent Stock Data Readings")
st.write(st_data.tail())

st.markdown('---')


st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'r')
st.pyplot(fig)

expander1 = st.expander('Interactive Chart', expanded = False)
with expander1:
  
  fig_exp1 = go.Figure()
  fig_exp1.add_trace(go.Scatter(x=st_data["Date"], y = st_data["Open"], name= "stock_open"))
  fig_exp1.add_trace(go.Scatter(x=st_data["Date"], y = st_data["Close"], name= "stock_close"))
  fig_exp1.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible = True)
  fig_exp1.update_xaxes(ticklabelposition="inside top", title='Time')
  fig_exp1.update_yaxes(ticklabelposition="inside top", title='Price')
  st.plotly_chart(fig_exp1)

st.markdown('---')


st.subheader('Closing price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close, 'g')
st.pyplot(fig)


st.markdown('---')

st.subheader('Closing price vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.markdown('---')

model = load_model("keras_model.h5")

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index= True)
input_data = scaler.fit_transform(final_df)

past_100_days_test = data_testing.tail(100)
input_30_data = scaler.fit_transform(past_100_days_test)
x_input= input_30_data.reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

#checkpoint 
lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
day_new=np.arange(1,101)
day_pred=np.arange(101,131)
inv_lst_output = scaler.inverse_transform(lst_output)
df1 = df.Close
lst_array = np.array(inv_lst_output)
lst_array1 = lst_array.ravel()

lst_series = pd.Series(lst_array1)
df_final = df1.append(lst_series, ignore_index = True)



df_30 = pd.DataFrame(lst_series)
#checkpoint 


x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])
  
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scaler_factor = 1/scaler[0]
y_predicted = y_predicted* scaler_factor
y_test = y_test* scaler_factor



st.subheader('Predictions vs Original')
fig1 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)


st.markdown('---')

st.subheader('30 Days Predictions')
columns = st.columns((1,4))
with columns[0]:
  st.text('Reading')
  st.write(df_30)

with columns[1]:
  st.text('Chart')
  fig2 = plt.figure(figsize=(12,6))
  plt.plot(df_30, 'b', label = 'Original Price')
  plt.legend()
  st.pyplot(fig2)


st.markdown('---')

st.subheader('Appended 30 Days Predictions Trend')
final_fig = plt.figure(figsize=(12,6))
plt.plot(df_final, 'r')
st.pyplot(final_fig)
