# %%
from vnpy.trader.optimize import OptimizationSetting
from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctabacktester import BacktesterEngine
from vnpy_ctastrategy.strategies.atr_rsi_strategy import (
    AtrRsiStrategy,
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


print(SETTINGS)

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
engine.set_parameters(
    vt_symbol=code,
    interval=interval,
    start=start,
    end=end,
    rate=0.3/10000,
    slippage=0.2,
    size=300,
    pricetick=0.2,
    capital=200000,
)
engine.add_strategy(AtrRsiStrategy, {})
engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()
