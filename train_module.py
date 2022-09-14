import tushare as ts
import math
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression

ts.set_token('51fd5a77415e6299ad8243e387472a5552e3d24f5c889781caef6d89')
df = ts.pro_bar(ts_code='600519.SH', adj='qfq', start_date='20170101', end_date='20220911')
df['trade_date'] = pd.to_datetime(df['trade_date'], format = "%Y/%m/%d %H:%M:%S")

df.set_index('trade_date', inplace=True)  # 设置索引覆盖原来的数据
df = df.sort_index(ascending=True)  # 将时间顺序升序，符合时间序列

# print(df)

#['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
df['hl_pct'] = (df['high'] - df['low']) / df['close'] * 100.0
df = df[['close', 'hl_pct', 'pct_chg', 'vol']]

#header = list(df)
#print(header)

# slice a dataframe with row index
# df = df.iloc[0:10]


forecast_col = 'close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
print(f'forecast_out {forecast_out}')

df['label'] = df[forecast_col].shift(-forecast_out)

print(df)

# shape: row, col
print(df.shape)

# Return the last n rows. default 5
print(df.tail())

#Remove rows or columns by specifying label names and corresponding axis, or by specifying directly index or column names. 
#axis{0 or ‘index’, 1 or ‘columns’}, default 0
X = df.drop(columns=['label'], axis = 1).to_numpy()  # to_numpy: convert to numpy array

# preprocessing normalize data for linear/logistic regression
X = preprocessing.scale(X)


X_lately = X[-forecast_out:]

date = np.array(df.index)

print(date[-forecast_out:])

X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

#Split arrays or matrices into random train and test subsets.
# X: data, Y: label
X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)


forecast_set = clf.predict(X_lately)

print(forecast_set,accuracy,forecast_out)
print("---")
print(forecast_set)
print("-x--")

"""
style.use('ggplot')

df['forecast']=np.nan



last_date = df.iloc[-1].name

last_unix = last_date.timestamp()
print(last_date,last_unix)

# sec in a day
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


print(df.tail())

df['close'].plot()
df['forecast'].plot()
plt.show()
"""


"""
svm
for k in ['linear','poly','rbf','sigmoid']:
    clf2 = svm.SVR(k)
    clf2.fit(X_train,y_train)
    accuracy2 = clf2.score(X_test,y_test)    
    print(accuracy2)
"""
"""
clf3 = svm.SVC(kernel='linear',C=1)
scores = cross_val_score(clf3,X,y,cv=5,scoring='f1_macro')
print(scores)
"""