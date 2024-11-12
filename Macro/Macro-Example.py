# Import necessary libraries
import pandas as pd  # For data manipulation
import yfinance as yf  # For retrieving financial data
import backtrader as bt  # For backtesting
from pandas_datareader import data as pdr  # Alternative data source
import datetime  # To handle date ranges

# Set the date range for our backtest
start_date = datetime.datetime(2010, 1, 1)  # Start of the backtest period
end_date = datetime.datetime.now()  # End of the backtest period

# Define the symbol for S&P 500 ETF (SPY is a popular ETF that tracks the S&P 500)
symbol = 'SPY'

# Fetch S&P 500 historical data using yfinance
data = yf.download(symbol, start=start_date, end=end_date)

# Load example macroeconomic indicators data
# Assume this data is pre-processed and saved in a CSV file named 'macro_indicators.csv'
# This file should contain indicators like interest rate, inflation, etc., indexed by date
macro_data = pd.read_csv('macro_indicators.csv', index_col='Date', parse_dates=True)

# Join S&P 500 price data with macroeconomic data
# This merges data on the date, forward-filling any missing macro data (assumes steady rates until change)
data = data.join(macro_data, how='inner').fillna(method='ffill')

# Backtrader Strategy Definition
class MacroFactorStrategy(bt.Strategy):
    # Assign weights to each factor; adjust these for different levels of influence on buy/sell signals
    params = (
        ('interest_rate_weight', -0.5),  # Weight for interest rate factor
        ('inflation_weight', -0.3),  # Weight for inflation factor
        ('gdp_weight', 0.7),  # Weight for GDP growth factor
        ('consumer_conf_weight', 0.5),  # Weight for consumer confidence factor
        ('credit_spread_weight', -0.4),  # Weight for credit spread factor
    )
    
    # Initialize strategy variables and indicators
    def __init__(self):
        # Reference to closing price of SPY (S&P 500 ETF)
        self.data_close = self.data.close

        # Calculate a single macro signal by summing up the weighted factors
        # A positive result signals a 'buy,' and a negative result signals a 'sell'
        self.macro_signal = (
            self.params.interest_rate_weight * self.data.interest_rate +
            self.params.inflation_weight * self.data.inflation +
            self.params.gdp_weight * self.data.gdp_growth +
            self.params.consumer_conf_weight * self.data.consumer_conf +
            self.params.credit_spread_weight * self.data.credit_spread
        )

    # Define logic to be applied at each step (every day of trading data)
    def next(self):
        # Check if the macro signal suggests buying
        if self.macro_signal > 0 and not self.position:  # Buy signal if no current position
            self.buy()
        # Check if the macro signal suggests selling
        elif self.macro_signal <= 0 and self.position:  # Sell signal if already holding position
            self.sell()

# Set up the Backtrader environment ('Cerebro') to run the backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(MacroFactorStrategy)  # Add the strategy we defined to Cerebro

# Custom class to ensure Backtrader can read our macroeconomic columns
# We inherit PandasData and add lines for each macro factor we are using
class PandasData(bt.feeds.PandasData):
    lines = ('interest_rate', 'inflation', 'gdp_growth', 'consumer_conf', 'credit_spread',)
    params = (('interest_rate', -1), ('inflation', -1), ('gdp_growth', -1), ('consumer_conf', -1), ('credit_spread', -1))

# Load the data feed into Backtrader using the custom PandasData class
feed = PandasData(dataname=data)  # Wrap our data in the custom feed class
cerebro.adddata(feed)  # Add the data feed to Cerebro

# Set initial cash for the backtest account
cerebro.broker.setcash(100000)  # Starting with $100,000 in cash

# Run the backtest
cerebro.run()

# Plot the results to visualize the strategy's performance
cerebro.plot()
