from pickletools import optimize
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

import tushare as ts
import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

stock_code = '600775.SH'
ts.set_token('51fd5a77415e6299ad8243e387472a5552e3d24f5c889781caef6d89')
data = ts.pro_bar(ts_code=stock_code, adj='qfq', start_date='20120101', end_date='20211231')
data['date'] = pd.to_datetime(data['trade_date'], format = "%Y/%m/%d %H:%M:%S")
data.set_index('date', inplace=True)  # 设置索引覆盖原来的数据
data = data.sort_index(ascending=True)  # 将时间顺序升序，符合时间序列
#data = data[['high', 'low', 'open', 'close', 'vol']]
print(data)
# ['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
"""
company = 'FB'

start = dt.datetime(2019, 1, 1)
end = dt.datetime(2019, 1, 5)
data = web.DataReader(company, 'yahoo', start, end)

print(list(data))
"""
"""
                  High         Low        Open       Close    Volume   Adj Close
Date                                                                            
2021-09-17  371.410004  361.589996  371.410004  364.720001  26299000  364.720001
2021-09-20  361.029999  349.799988  359.299988  355.700012  19822800  355.700012
2021-09-21  360.040009  355.190002  358.500000  357.480011  11751900  357.480011
"""

# prepare data
# scale down the 
scaler = MinMaxScaler(feature_range=(0,1))
before_scale = data['close'].values.reshape(-1, 1)
print(before_scale)
scaled_data = scaler.fit_transform(before_scale)

prediction_days = 60
print(scaled_data)

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days: x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

exit(0)
# build the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


test_data = ts.pro_bar(ts_code=stock_code, adj='qfq', start_date='20220101')
test_data['date'] = pd.to_datetime(test_data['trade_date'], format = "%Y/%m/%d %H:%M:%S")
test_data.set_index('date', inplace=True)  # 设置索引覆盖原来的数据
test_data = test_data.sort_index(ascending=True)  # 将时间顺序升序，符合时间序列
#test_data = test_data[['high', 'low', 'open', 'close', 'vol']]
actual_prices = test_data['close']

total_dataset = pd.concat((data['close'], test_data['close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
print(predicted_prices)
print(type(predicted_prices))
print(actual_prices.values)
print(type(actual_prices.values))

plt.plot(actual_prices.values, color='black', label=f'Actual price')
plt.plot(predicted_prices, color='green', label='Predicted price')
plt.title('Share Price')
plt.xlabel('Time')
plt.ylabel('share price')
plt.legend()
plt.show()
