import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta, date

def online_learning(model, new_data, scaler):
    scaled_new_data = scaler.transform(new_data)
    x_new = scaled_new_data[:-1].reshape(1, -1, 1)
    y_new = scaled_new_data[-1].reshape(1, 1)
    
    prediction = model.predict(x_new)
    
    model.fit(x_new, y_new, epochs=1, verbose=0)
    
    return scaler.inverse_transform(prediction)[0][0]

def update_data_and_model(stock, model, scaler, last_100_days):
    end = datetime.now()
    start = end - timedelta(days=1)
    new_data = yf.download(stock, start, end)
    if not new_data.empty:
        new_close = new_data['Adj Close'].values[-1]
        last_100_days = np.append(last_100_days[1:], new_close)
        
        predicted_price = online_learning(model, last_100_days.reshape(-1, 1), scaler)
        
        st.write(f"New data point: {new_close}")
        st.write(f"Predicted next price: {predicted_price}")
    
    return last_100_days

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)

model = load_model(r"C:\Users\incha\stock_price_prediction_project\Latest_stcok_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

st.subheader("Future Close Price values")

def predict_future_stock(no_of_days, prev_100):
    future_predictions = []
    prev_100 = scaler.fit_transform(prev_100['Adj Close'].values.reshape(-1,1)).reshape(1,-1,1)
    for _ in range(no_of_days):
        next_day = model.predict(prev_100)[0, 0]  
        future_predictions.append(next_day)
        prev_100 = np.append(prev_100[:, 1:, :], [[[next_day]]], axis=1)  
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

no_of_days = int(st.text_input("Enter the number of days to be predicted from current date : ", "10"))  
future_results = predict_future_stock(no_of_days, prev_100= google_data[['Adj Close']].tail(100))
print(future_results)

future_results = np.array(future_results).reshape(-1,1)
fig = plt.figure(figsize = (15,5))
plt.plot(pd.DataFrame(future_results), marker = 'o')
for i in range(len(future_results)):
    plt.text(i, future_results[i], int(future_results[i][0]))
plt.xlabel('Future days')
plt.ylabel('Close Price')
plt.title("Future Close price of stock")
st.pyplot(fig)

if st.button("Perform Online Learning"):
    last_100_days = google_data['Adj Close'].tail(100).values
    last_100_days = update_data_and_model(stock, model, scaler, last_100_days)
    st.write("Model updated with the latest data point.")