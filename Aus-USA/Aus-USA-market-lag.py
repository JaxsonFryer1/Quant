import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import matplotlib.pyplot as plt

# Define the date range
start_date = '2020-11-01'
end_date = '2024-11-27'

# Download data for U.S. S&P 500 and Australian ETF
spx = yf.download('^GSPC', start=start_date, end=end_date)
ewa = yf.download('EWA', start=start_date, end=end_date)



# Clean and access SPX data
if isinstance(spx.columns, pd.MultiIndex):
    spx.columns = spx.columns.map('_'.join)  # Flatten MultiIndex
if 'Close_^GSPC' in spx.columns:
    spx = spx[['Close_^GSPC']].copy()
    spx.rename(columns={'Close_^GSPC': 'Close'}, inplace=True)
else:
    raise KeyError("Unable to locate 'Close_^GSPC' column in the SPX DataFrame")

# Clean and access EWA data
if isinstance(ewa.columns, pd.MultiIndex):
    ewa.columns = ewa.columns.map('_'.join)  # Flatten MultiIndex
if 'Open_EWA' in ewa.columns and 'Close_EWA' in ewa.columns:
    ewa = ewa[['Open_EWA', 'Close_EWA']].copy()
    ewa.rename(columns={'Open_EWA': 'Open', 'Close_EWA': 'Close'}, inplace=True)
else:
    raise KeyError("Unable to locate 'Open_EWA' or 'Close_EWA' columns in the EWA DataFrame")

# Add shifted close price for SPX to calculate returns
spx['Prev_Close'] = spx['Close'].shift(1)
spx['Return'] = (spx['Close'] - spx['Prev_Close']) / spx['Prev_Close']

# Align SPX and EWA data by date
common_dates = spx.index.intersection(ewa.index)
spx = spx.loc[common_dates]
ewa = ewa.loc[common_dates]

# Merge SPX and EWA into a single DataFrame
data = pd.DataFrame({
    'SPX_Return': spx['Return'],
    'EWA_Open': ewa['Open'],
    'EWA_Close': ewa['Close'],
    'SPX_Close': spx['Close']
}).dropna()

# Initialize trading signals and portfolio
data['Signal'] = 0
data['Position'] = 0
data['Portfolio_Value'] = 0

# Parameters
initial_capital = 100000
capital = initial_capital
position_size = 0
stop_loss_pct = 0.12  # 12% trailing stop-loss
high_price = 0  # Track the highest price while the position is open
highest_open = -1  # Track the highest open price seen after buying

# Generate buy signals when SPX return > 1%
data.loc[data['SPX_Return'] > 0.01, 'Signal'] = 1

# Backtesting logic
portfolio_values = []
for i in range(len(data)):
    signal = data['Signal'].iloc[i]
    open_price = data['EWA_Open'].iloc[i]
    close_price = data['EWA_Close'].iloc[i]
    
    # Entry logic
    if signal == 1 and position_size == 0:
        position_size = capital / open_price
        highest_open = max(highest_open, open_price)  # Update highest open seen
        high_price = highest_open  # Reset high price
        capital = 0
        print(f"Buy on {data.index[i].date()} at {open_price:.2f}")
    
    # Exit logic
    if position_size > 0:
        # Update the high price while the position is open
        high_price = max(high_price, close_price)
        
        # Trailing stop-loss condition
        if (close_price - high_price) / high_price <= -stop_loss_pct:
            capital = position_size * close_price
            position_size = 0
            print(f"Trailing stop-loss triggered on {data.index[i].date()} at {close_price:.2f}")
        
        # Exit at close if there's no buy signal
        elif signal == 0:
            capital = position_size * close_price
            position_size = 0
            print(f"Sell on {data.index[i].date()} at {close_price:.2f}")
    
    # Portfolio value
    if position_size > 0:
        portfolio_value = position_size * close_price
    else:
        portfolio_value = capital
    portfolio_values.append(portfolio_value)

# Add portfolio values to the data
data['Portfolio_Value'] = portfolio_values

# Add SPX normalized values and EWA normalized values for comparison
data['SPX_Normalized'] = data['SPX_Close'] / data['SPX_Close'].iloc[0] * initial_capital
data['EWA_Normalized'] = data['EWA_Close'] / data['EWA_Close'].iloc[0] * initial_capital

# Performance Metrics
total_return = (data['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital
num_years = (data.index[-1] - data.index[0]).days / 365.25
cagr = (data['Portfolio_Value'].iloc[-1] / initial_capital) ** (1 / num_years) - 1
rolling_max = data['Portfolio_Value'].cummax()
drawdown = (data['Portfolio_Value'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# Compute Sharpe Ratio
daily_returns = data['Portfolio_Value'].pct_change().dropna()
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

# Output performance metrics
print(f"Total Return: {total_return:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Plot portfolio performance with SPX and EWA
plt.figure(figsize=(12, 6))
plt.plot(data['Portfolio_Value'], label='Portfolio Value')
plt.plot(data['SPX_Normalized'], label='SPX Normalized')
plt.plot(data['EWA_Normalized'], label='EWA Normalized')
plt.title('Portfolio Value vs. SPX and EWA')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
