import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize the REST API
api = REST(API_KEY, SECRET_KEY, base_url='https://paper-api.alpaca.markets')



# Check account details
account = api.get_account()
print(f"Account Status: {account.status}")
print(f"Equity: {account.equity}")
print(f"Buying Power: {account.buying_power}")

warnings.filterwarnings('ignore')

# Define the date range
start_date = '2014-01-01'
end_date = '2024-11-21'

# Helper functions
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def compute_ATR(data, high_col, low_col, close_col, period=14):
    data['H-L'] = data[high_col] - data[low_col]
    data['H-PC'] = abs(data[high_col] - data[close_col].shift(1))
    data['L-PC'] = abs(data[low_col] - data[close_col].shift(1))
    tr = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    ATR = tr.rolling(window=period).mean()
    data.drop(['H-L', 'H-PC', 'L-PC'], axis=1, inplace=True)  # Clean up intermediate columns
    return ATR

# Download data for U.S. indicators
symbols = {
    'SPX': '^GSPC', 'VIX': '^VIX', 'USO': 'USO', 
    'GLD': 'GLD', 'XLK': 'XLK', 'XLF': 'XLF'
}

us_data = {}
for name, ticker in symbols.items():
    df = yf.download(ticker, start=start_date, end=end_date)
    df.rename(columns=lambda x: f"{name}_{x}", inplace=True)
    us_data[name] = df

# Download data for Australian ETFs
aus_symbols = {'EWA': 'EWA', 'OZR': 'OZR.AX', 'ATEC': 'ATEC.AX', 'BBOZ': 'BBOZ.AX'}

aus_data = {}
for name, ticker in aus_symbols.items():
    df = yf.download(ticker, start=start_date, end=end_date)
    df.rename(columns=lambda x: f"{name}_{x}", inplace=True)
    aus_data[name] = df

# Print columns for all datasets
print("U.S. Data Columns:")
for name, df in us_data.items():
    print(f"{name}: {df.columns.tolist()}")

print("\nAustralian Data Columns:")
for name, df in aus_data.items():
    print(f"{name}: {df.columns.tolist()}")

# Merge data into a single DataFrame
us_merged = pd.concat([df for df in us_data.values()], axis=1)
aus_merged = pd.concat([df for df in aus_data.values()], axis=1)
data = pd.concat([us_merged, aus_merged], axis=1)
data.dropna(inplace=True)

# Flatten column names for easier access
data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in data.columns]

# Confirm the exact names for SPX_Close and EWA_Close
spx_close_col = 'SPX_Close'
ewa_close_col = 'EWA_Close'

# Adjust column names for easier access
data.rename(columns={
    'SPX_Close_SPX_^GSPC': 'SPX_Close',
    'SPX_High_SPX_^GSPC': 'SPX_High',
    'SPX_Low_SPX_^GSPC': 'SPX_Low',
    'EWA_Close_EWA_EWA': 'EWA_Close',
    'EWA_Open_EWA_EWA': 'EWA_Open',
    'EWA_High_EWA_EWA': 'EWA_High',
    'EWA_Low_EWA_EWA': 'EWA_Low'
}, inplace=True)

# Ensure these columns exist in the DataFrame
if spx_close_col not in data.columns or ewa_close_col not in data.columns:
    raise KeyError("SPX_Close or EWA_Close columns are missing in the dataset.")

# Calculate SPX returns
data['SPX_Return'] = data[spx_close_col].pct_change()

# Calculate ATR for SPX and EWA
data['SPX_ATR'] = compute_ATR(data.copy(), 'SPX_High', 'SPX_Low', 'SPX_Close')
data['EWA_ATR'] = compute_ATR(data.copy(), 'EWA_High', 'EWA_Low', 'EWA_Close')

# Feature engineering
data['VIX_Return'] = data['VIX_Close_VIX_^VIX'].pct_change()
data['USO_Return'] = data['USO_Close_USO_USO'].pct_change()
data['GLD_Return'] = data['GLD_Close_GLD_GLD'].pct_change()
data['XLK_Return'] = data['XLK_Close_XLK_XLK'].pct_change()
data['XLF_Return'] = data['XLF_Close_XLF_XLF'].pct_change()

data['EWA_RSI'] = compute_RSI(data[ewa_close_col])
data['Rolling_Corr'] = data[spx_close_col].rolling(window=60).corr(data[ewa_close_col])

# Shift U.S. data to align with the Australian market
shift_columns = [col for col in data.columns if any(ind in col for ind in ['SPX_', 'VIX_', 'USO_', 'GLD_', 'XLK_', 'XLF_'])]
data[shift_columns] = data[shift_columns].shift(1)
data.dropna(inplace=True)

# Composite signal
data['Composite_Signal'] = (
    0.5 * (data['SPX_Return'] / data['SPX_Return'].std()) -
    0.3 * (data['VIX_Return'] / data['VIX_Return'].std()) +
    0.2 * (data['USO_Return'] / data['USO_Return'].std())
)

# Technical indicators
data['EWA_MA20'] = data[ewa_close_col].rolling(window=20).mean()
data['EWA_MA50'] = data[ewa_close_col].rolling(window=50).mean()
data['MA_Crossover'] = np.where(data['EWA_MA20'] > data['EWA_MA50'], 1, -1)

# Prepare features and target for machine learning
features = ['SPX_Return', 'VIX_Return', 'USO_Return', 'GLD_Return', 'XLK_Return', 'XLF_Return', 'EWA_RSI', 'MA_Crossover']
X = data[features]
y = np.where(data[ewa_close_col].shift(-1) > data[ewa_close_col], 1, 0)

# Ensure index is a DatetimeIndex
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)

# Debug index range
print(f"Data index range: {data.index.min()} to {data.index.max()}")

# Adjust split_date if it's outside the range
split_date = pd.Timestamp('2020-01-01')
if split_date < data.index.min() or split_date > data.index.max():
    split_date = data.index.min() + (data.index.max() - data.index.min()) / 2  # Midpoint of range
    print(f"Adjusted split_date to midpoint: {split_date}")

# Validate split
print(f"Using split_date: {split_date}")
print(f"Rows before split_date: {len(data[data.index < split_date])}")
print(f"Rows after split_date: {len(data[data.index >= split_date])}")

# Train-test split
X_train = X[data.index < split_date]
X_test = X[data.index >= split_date]
y_train = y[data.index < split_date]
y_test = y[data.index >= split_date]

# Check if training data is empty
if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("Training data is empty. Check the split_date or data filtering steps.")

# Fit the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
data['Predicted_Signal'] = model.predict(X)

# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=features)
print("Feature Importances:")
print(feature_importances.sort_values(ascending=False))

# Backtesting with risk management
initial_capital = 30000
data['Position'] = 0
data['Cash'] = initial_capital
data['Total'] = initial_capital
data['Holdings'] = 0
data['Entry_Price'] = np.nan

# Risk management parameters
stop_loss_pct = 0.05  # 5% stop loss
take_profit_pct = 1.0  # 100% take profit
max_position_size =  initial_capital  # Maximum position size is initial capital

for i in range(1, len(data)):
    prev_index = data.index[i - 1]
    index = data.index[i]
    
    if data.loc[prev_index, 'Predicted_Signal'] == 1 and data.loc[prev_index, 'Position'] == 0:
        # Buy signal and currently not in position
        # Implement position sizing based on ATR
        atr = data.loc[index, 'EWA_ATR']
        if pd.isna(atr) or atr == 0:
            atr = data['EWA_ATR'].mean()  # Use average ATR if current ATR is not available
        risk_per_share = atr
        risk_per_trade = 0.05 * data.loc[prev_index, 'Total']  # Risk 10% of current equity
        shares = risk_per_trade // risk_per_share
        # Limit position size
        max_shares = max_position_size // data.loc[index, 'EWA_Open']
        shares = min(shares, max_shares)
        # Ensure shares is at least 1
        shares = max(shares, 1)
        data.loc[index, 'Position'] = shares
        data.loc[index, 'Cash'] = data.loc[prev_index, 'Cash'] - (shares * data.loc[index, 'EWA_Open'])
        data.loc[index, 'Entry_Price'] = data.loc[index, 'EWA_Open']
        
    elif data.loc[prev_index, 'Position'] > 0:
        # Currently in position
        # Check for stop-loss or take-profit
        entry_price = data.loc[prev_index, 'Entry_Price']
        current_low = data.loc[index, 'EWA_Low']
        current_high = data.loc[index, 'EWA_High']
        shares = data.loc[prev_index, 'Position']
        # Check stop-loss
        if (current_low - entry_price) / entry_price <= -stop_loss_pct:
            # Exit at stop-loss price
            exit_price = entry_price * (1 - stop_loss_pct)
            data.loc[index, 'Cash'] = data.loc[prev_index, 'Cash'] + (shares * exit_price)
            data.loc[index, 'Position'] = 0
            data.loc[index, 'Entry_Price'] = np.nan
        # Check take-profit
        elif (current_high - entry_price) / entry_price >= take_profit_pct:
            # Exit at take-profit price
            exit_price = entry_price * (1 + take_profit_pct)
            data.loc[index, 'Cash'] = data.loc[prev_index, 'Cash'] + (shares * exit_price)
            data.loc[index, 'Position'] = 0
            data.loc[index, 'Entry_Price'] = np.nan
        # Check sell signal
        elif data.loc[prev_index, 'Predicted_Signal'] == 0:
            # Exit at current open price
            data.loc[index, 'Cash'] = data.loc[prev_index, 'Cash'] + (shares * data.loc[index, 'EWA_Open'])
            data.loc[index, 'Position'] = 0
            data.loc[index, 'Entry_Price'] = np.nan
        else:
            # Hold position
            data.loc[index, 'Position'] = shares
            data.loc[index, 'Cash'] = data.loc[prev_index, 'Cash']
            data.loc[index, 'Entry_Price'] = entry_price
    else:
        # Not in position and no buy signal
        data.loc[index, 'Position'] = 0
        data.loc[index, 'Cash'] = data.loc[prev_index, 'Cash']
        data.loc[index, 'Entry_Price'] = np.nan

    data.loc[index, 'Holdings'] = data.loc[index, 'Position'] * data.loc[index, 'EWA_Close']
    data.loc[index, 'Total'] = data.loc[index, 'Cash'] + data.loc[index, 'Holdings']

# Calculate cumulative returns and performance metrics
data['Cumulative_Returns'] = (data['Total'] / initial_capital)
returns = data['Total'].pct_change()
annualized_return = returns.mean() * 252
num_years = (data.index[-1] - data.index[0]).days / 365.25

annualized_volatility = returns.std() * np.sqrt(252)
r_f = 0.02 #2% risk free
sharpe_ratio = (annualized_return - r_f) / annualized_volatility

# Calculate drawdown
def calculate_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

data['Drawdown'], max_drawdown = calculate_drawdown(data['Total'])
print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Annualized return: {annualized_return}")


# Visualization of Portfolio Value and Drawdown
plt.figure(figsize=(14, 7))
plt.plot(data['Total'], label='Strategy Portfolio Value')
plt.plot(data['EWA_Close'] / data['EWA_Close'].iloc[0] * initial_capital, label='EWA Buy and Hold')
plt.plot(data['SPX_Close'] / data['SPX_Close'].iloc[0] * initial_capital, label='SPX Buy and Hold')
plt.title('Portfolio Value vs. Buy-and-Hold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()


