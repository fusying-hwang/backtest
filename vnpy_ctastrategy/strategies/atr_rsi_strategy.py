from time import sleep
from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)

from pickletools import optimize
from statistics import mode
import numpy as np
import pandas as pd
import datetime as dt
import tushare as ts

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

def inverse_predictions(predictions,scaler,prediction_index=2):
    '''This function uses the fitted scaler to inverse predictions, 
    the index should be set to the position of the target variable'''
    
    max_val = scaler.data_max_[prediction_index]
    min_val = scaler.data_min_[prediction_index]
    original_values = (predictions*(max_val - min_val )) + min_val
    
    return original_values

class AtrRsiStrategy(CtaTemplate):
    """"""

    author = "用Python的交易员"

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar)
        self.prediction_days = 7
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.model = keras.models.load_model('/Users/chenchen/backtest/ningdeshidai.h5')
        self.fixed_size = 100 #一手
        stock_code = '300750.SZ'
        self.capital = 200000.0
        features = ['open', 'high', 'low', 'close', 'vol', 'amount']
        ts.set_token('51fd5a77415e6299ad8243e387472a5552e3d24f5c889781caef6d89')
        data = ts.pro_bar(ts_code=stock_code, adj='qfq', start_date='20120101', end_date='20211231')
        data['date'] = pd.to_datetime(data['trade_date'], format = "%Y/%m/%d %H:%M:%S")
        data.set_index('date', inplace=True)  # 设置索引覆盖原来的数据
        data = data.sort_index(ascending=True)  # 将时间顺序升序，符合时间序列
        raw_data = data[features]
        before_scale = raw_data.values.reshape(-1, len(features))
        self.scaler.fit_transform(before_scale)
        print("scaler fitted")

        self.begin_backtest = False

        self.cur_date : dt.datetime = None
        self.cur_bar : BarData = None
        self.real_data = []


    def on_init(self):
        """
        Callback when strategy is inited.
        """
        print("策略初始化")

        # set the pre-requested time
        self.load_bar(7)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.begin_backtest = True
        print("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        print("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        print(f'on_tick {tick}')
        self.bg.update_tick(tick)

    def try_compose_bar(self, bar: BarData, is_new_trade_day: bool):
        if is_new_trade_day:
            self.cur_bar = bar
        else:
            self.cur_bar.volume += bar.volume
            self.cur_bar.turnover += bar.turnover
            self.cur_bar.high_price = max(self.cur_bar.high_price, bar.high_price)
            self.cur_bar.low_price = min(self.cur_bar.low_price, bar.low_price)
            self.cur_bar.close_price = bar.close_price
        
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        # 目的是清空未成交的委托(包括本地停止单，限价单)，保证当前时间点委托状态是唯一的。
        self.cancel_all()
        #print(bar)

        if bar.datetime.date() != self.cur_date.date() and self.cur_date is not None:
            self.real_data.append([self.cur_bar.open_price, self.cur_bar.high_price, self.cur_bar.low_price, self.cur_bar.close_price, self.cur_bar.volume, self.cur_bar.turnover])
            self.try_compose_bar(bar, True)
        else:
            if self.begin_backtest:
                if bar.datetime.time() == dt.datetime(hour=14, minute=30):
                    self.real_data = self.real_data[-self.prediction_days + 1:]
                    pre_feature = self.real_data + [[self.cur_bar.open_price, self.cur_bar.high_price, self.cur_bar.low_price, self.cur_bar.close_price, self.cur_bar.volume, self.cur_bar.turnover]]
                    pre_feature = np.array(pre_feature)
                    pre_feature = np.reshape(pre_feature, (pre_feature.shape[0], pre_feature.shape[1], len(self.features)))
                    predict_tomorrow = self.model.predict(pre_feature)
                    predict_tomorrow = inverse_predictions(predict_tomorrow, self.scaler, 0)
                    predicted_price = predict_tomorrow[0][0]
                    if predicted_price > bar.close_price and self.capital >= bar.close_price * self.fixed_size:
                        self.buy(bar.close_price, self.fixed_size)
                    else:
                        if self.pos > 0:
                            self.sell(bar.close_price, self.pos, stop=True)

            self.cur_date = bar.datetime.date()
            self.try_compose_bar(bar, False)
        self.put_event()           

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        print(f'new order {order}')

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        # todo
        print(f'on trade {trade}')

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        print(f'on stop order {stop_order}')
