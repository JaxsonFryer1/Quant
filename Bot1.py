from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps,resample_apply, barssince
import talib
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from backtesting.test import SMA, GOOG

def optim(series):
     if series["# Trades"] < 10:
          return -1
     return series["Equity Final [$]"]/ series["Exposure Time [%]"]

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()

class RsiOscillator(Strategy):
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14 
    def init(self):
        self.rsi_daily = self.I(talib.RSI, self.data.Close, self.rsi_window)
        self.rsi_weekly = resample_apply(
            "W-FRI",
            talib.RSI,
            self.data.Close,
            self.rsi_window
        )

    def next(self):
        price = self.data.Close[-1]
        if crossover(self.rsi_daily, self.upper_bound) : # and self.rsi_weekly[-1] > self.upper_bound       and barssince(self.rsi_daily > self.upper_bound) == 3
            # print(self.position.size)
            # print(self.position.pl_pct)
            self.position.close()
            
        elif crossover(self.lower_bound,self.rsi_daily): # and self.rsi_weekly[-1] < self.lower_bound
            # self.buy(tp=1.15*price, sl=0.95*price, size = 0.15)
            self.buy()

bt = Backtest(GOOG, RsiOscillator, cash=10_000)
stats,heatmap = bt.optimize(
    upper_bound = range(50,85,5),
    lower_bound = range(10,45,5),
    rsi_window = range(10,30,5),
    maximize = "Equity Final [$]",
    constraint = lambda param: param.upper_bound > param.lower_bound,
    return_heatmap=True)
bt.plot()
print(stats)

print(stats['_trades'].to_string())




#matplotlib heat map
# hm = heatmap.groupby(["upper_bound", "lower_bound"]).mean().unstack()
# print(hm)
# sns.heatmap(hm)
# plt.show()
# print(hm)

#backtest heatmap and plotting
# plot_heatmaps(heatmap, agg="mean")
# lower_bound = stats["_strategy"].lower_bound
# upper_bound = stats["_strategy"].upper_bound
# bt.plot(filename=f"plots/RSIOscillator-{lower_bound}-{upper_bound}")


#optimization
# stats,heatmap = bt.optimize(
#     upper_bound = range(50,85,5),
#     lower_bound = range(10,45,5),
#     rsi_window = 14,
#     maximize = "Sharpe Ratio",
#     constraint = lambda param: param.upper_bound > param.lower_bound,
#     return_heatmap=True)