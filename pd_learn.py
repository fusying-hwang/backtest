#!/usr/bin/env python

import numpy as np
import pandas as pd


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

from tensorflow import keras

stock_code = '300750.SZ'

features = ['open', 'high', 'low', 'close', 'vol', 'amount']

prediction_days = 7
scaler = MinMaxScaler(feature_range=(0,1))

model = keras.models.load_model('ningdeshidai.h5')


ts.set_token('51fd5a77415e6299ad8243e387472a5552e3d24f5c889781caef6d89')
data = ts.pro_bar(ts_code=stock_code, adj='qfq', start_date='20120101', end_date='20211231')
data['date'] = pd.to_datetime(data['trade_date'], format = "%Y/%m/%d %H:%M:%S")
data.set_index('date', inplace=True)  # 设置索引覆盖原来的数据
data = data.sort_index(ascending=True)  # 将时间顺序升序，符合时间序列
raw_data = data[features]

test_data = ts.pro_bar(ts_code=stock_code, adj='qfq', start_date='20220101')
test_data['date'] = pd.to_datetime(test_data['trade_date'], format = "%Y/%m/%d %H:%M:%S")
test_data.set_index('date', inplace=True)  # 设置索引覆盖原来的数据
test_data = test_data.sort_index(ascending=True)  # 将时间顺序升序，符合时间序列
#test_data = test_data[['high', 'low', 'open', 'close', 'vol']]
actual_prices = test_data['close']

def inverse_predictions(predictions,scaler,prediction_index=2):
    '''This function uses the fitted scaler to inverse predictions, 
    the index should be set to the position of the target variable'''
    
    max_val = scaler.data_max_[prediction_index]
    min_val = scaler.data_min_[prediction_index]
    print(f'max_val {max_val}, min_val {min_val}')
    original_values = (predictions*(max_val - min_val )) + min_val
    
    return original_values

before_scale = raw_data.values.reshape(-1, len(features))

scaled_data = scaler.fit_transform(before_scale)

total_dataset = pd.concat((raw_data, test_data[features]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, len(features))
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, :])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(features)))

predicted_prices = model.predict(x_test)
print(predicted_prices.shape)
#predicted_prices = predicted_prices.reshape(-1, 1)
#predicted_prices = scaler.inverse_transform(predicted_prices)
predicted_prices = inverse_predictions(predicted_prices, scaler, 0)
print(predicted_prices)
print(type(predicted_prices))
actual_prices = test_data['close']
print(actual_prices.values)
print(type(actual_prices.values))

# create a dataframe and use the index from another dataframe
predicted_prices = pd.DataFrame(predicted_prices, index = actual_prices.index)
plt.plot(actual_prices, color='black', label=f'Actual price')
plt.plot(predicted_prices, color='green', label='Predicted price')
plt.title('Share Price')
plt.xlabel('Time')
plt.ylabel('share price')
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()

"""
I named this file pandas.py and it imported pandas so it gonna import itself and the code looks like running twice
"""


np.random.seed(100)
r = np.arange(80)
r =r.reshape((8, 10))
r = r.reshape((-1, 1))
r = r.reshape((-1, 8))

# negetive number is just like python list index -1 means the last index
#r = r.reshape((-2, 4))
print(r)

print(r[1:3])
print(r[1:3, 0:3])

print(r[1: 3, 0: 1])
# 退化成一个数组
print(r[1:3, 0])

r = r.reshape(-1, 1)
print(r[0: 4])
print(r[0: 4, 0])
print(r[0, 0])
