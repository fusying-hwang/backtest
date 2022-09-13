import tushare as ts


ts.set_token('51fd5a77415e6299ad8243e387472a5552e3d24f5c889781caef6d89')
df = ts.pro_bar(ts_code='600519.SH', adj='qfq', start_date='20190101', end_date='20220911')

#df = df[['close', 'pre_close']]
print(df)
header = list(df)
print(header)
