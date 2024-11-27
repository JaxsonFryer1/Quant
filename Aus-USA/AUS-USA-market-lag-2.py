import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# Define the date range
start_date = '2015-01-01'
end_date = '2023-10-01'

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
spx_close_col = 'SPX_Close_SPX_^GSPC'
ewa_close_col = 'EWA_Close_EWA_EWA'

# Ensure these columns exist in the DataFrame
if spx_close_col not in data.columns or ewa_close_col not in data.columns:
    raise KeyError("SPX_Close or EWA_Close columns are missing in the dataset.")

# SPX columns
spx_high_col = 'SPX_High_SPX_^GSPC'
spx_low_col = 'SPX_Low_SPX_^GSPC'

# EWA columns
ewa_high_col = 'EWA_High_EWA_EWA'
ewa_low_col = 'EWA_Low_EWA_EWA'

# Calculate SPX returns
data['SPX_Return'] = data[spx_close_col].pct_change()

# Calculate ATR for SPX
data['SPX_ATR'] = compute_ATR(data, spx_high_col, spx_low_col, spx_close_col)

# Calculate ATR for EWA
data['EWA_ATR'] = compute_ATR(data, ewa_high_col, ewa_low_col, ewa_close_col)

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



# Backtesting
initial_capital = 100000
data['Position'] = 0
data['Cash'] = initial_capital
data['Total'] = initial_capital
data['Holdings'] = 0

for i in range(1, len(data)):
    prev_index = data.index[i - 1]
    index = data.index[i]
    if data.loc[prev_index, 'Predicted_Signal'] == 1 and data.loc[prev_index, 'Position'] == 0:
        shares = initial_capital // data.loc[index, 'EWA_Open_EWA_EWA']
        data.loc[index, 'Position'] = shares
        data.loc[index, 'Cash'] = data.loc[prev_index, 'Cash'] - (shares * data.loc[index, 'EWA_Open_EWA_EWA'])
    elif data.loc[prev_index, 'Predicted_Signal'] == 0 and data.loc[prev_index, 'Position'] > 0:
        shares = data.loc[prev_index, 'Position']
        data.loc[index, 'Cash'] = data.loc[prev_index, 'Cash'] + (shares * data.loc[index, 'EWA_Open_EWA_EWA'])
        data.loc[index, 'Position'] = 0
    else:
        data.loc[index, 'Position'] = data.loc[prev_index, 'Position']
        data.loc[index, 'Cash'] = data.loc[prev_index, 'Cash']

    data.loc[index, 'Holdings'] = data.loc[index, 'Position'] * data.loc[index, ewa_close_col]
    data.loc[index, 'Total'] = data.loc[index, 'Cash'] + data.loc[index, 'Holdings']

data['Cumulative_Returns'] = (1 + data['Total'].pct_change()).cumprod()
annualized_return = data['Total'].pct_change().mean() * 252
annualized_volatility = data['Total'].pct_change().std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(data['Total'], label='Strategy Portfolio Value')
plt.plot(data[ewa_close_col] / data[ewa_close_col].iloc[0] * initial_capital, label='EWA Buy and Hold')
plt.title('Portfolio Value vs. Buy-and-Hold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
