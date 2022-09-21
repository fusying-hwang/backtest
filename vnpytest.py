# %%
from vnpy.trader import setting
from vnpy.trader.optimize import OptimizationSetting
from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctabacktester import BacktesterEngine
from vnpy_ctastrategy.strategies.lstm_strategy import (
    LongShortTermMemoryStrategy,
)
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine, LogEngine
from datetime import datetime
from vnpy.trader.setting import SETTINGS
from sklearn import datasets

SETTINGS["datafeed.name"] = "tushare"

SETTINGS["datafeed.username"] = "token"
SETTINGS['datafeed.password'] = "51fd5a77415e6299ad8243e387472a5552e3d24f5c889781caef6d89"
SETTINGS['database.timezone'] = 'Asia/Shanghai'
SETTINGS['database.name'] = 'sqlite'
SETTINGS['database.database'] = 'database.db'
SETTINGS['database.host'] = 'localhost'
SETTINGS['database.port'] = 3306
SETTINGS['database.user'] = 'root'

SETTINGS['log.level'] = 10

event_engine = EventEngine()

main_engine = MainEngine(event_engine)
log_engine = LogEngine(main_engine=main_engine, event_engine=event_engine)

engine = BacktesterEngine(main_engine=main_engine, event_engine=event_engine)

code = "300750.SZSE"
interval = "30m"
start = datetime(2021, 12, 20)
end = datetime(2022, 9, 20)

engine.run_downloading(vt_symbol=code,
                       interval=interval,
                       start=start,
                       end=end)

engine = BacktestingEngine()

"""
设置回测的参数。参数及其含义如下 1. vt_symbol ==> 产品名称 2. interval ==> 周期 3. start ==> 开始时间 4. rate ==> 手续费 5. 
slippage ==> 滑点 6. size ==> 合约乘数 7. pricetick ==> 价格跳动 8. capital ==> 回测资本 9. end ==> 截止时间 
10. mode ==> 回测的模式, 一共有两种:BacktestingMode.BAR和BacktestingMode.TICK 11. inverse ==> 周期
"""

settings = {
    'vt_symbol': code,
    'interval':interval,
    'start':start,
    'end':end,
    'rate':0.3/10000,
    'slippage':0.2,
    'size':100,  # 设置一手股票个数
    'pricetick':0.2,
    'capital':200000.0, 
}

engine.set_parameters(
    **settings
)
engine.add_strategy(LongShortTermMemoryStrategy, settings)
engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()
