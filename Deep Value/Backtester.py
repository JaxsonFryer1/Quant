import os
import pandas as pd
import matplotlib.pyplot as plt

def backtest_strategy(tickers, folder_path="financial_data", start_date=None, end_date=None):
    """
    Backtest a strategy using historical stock data.

    Args:
        tickers (list): List of stock tickers.
        folder_path (str): Path to stored financial data.
        start_date (str): Start date for backtesting (YYYY-MM-DD).
        end_date (str): End date for backtesting (YYYY-MM-DD).

    Returns:
        pd.DataFrame: Portfolio value over time.
    """
    portfolio = pd.DataFrame()

    # Convert input dates to pd.Timestamp (tz-naive)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    for ticker in tickers:
        try:
            print(f"Backtesting {ticker}...")
            # Load historical data
            history_path = os.path.join(folder_path, f"{ticker}_history.csv")
            if not os.path.exists(history_path):
                print(f"File for {ticker} not found: {history_path}")
                continue
            
            history = pd.read_csv(history_path, index_col=0, parse_dates=True)

            # Ensure index is tz-naive
            if hasattr(history.index, 'tz'):
                history.index = history.index.tz_localize(None)

            # Filter by date range
            history = history.loc[start_date:end_date]

            # Ensure 'Close' column exists
            if 'Close' not in history.columns:
                print(f"'Close' column missing in {ticker} data.")
                continue

            # Add to portfolio
            portfolio[ticker] = history['Close']
        except Exception as e:
            print(f"Failed to load data for {ticker}: {e}")
    
    if portfolio.empty:
        print("No valid data loaded for portfolio.")
        return portfolio

    # Calculate total portfolio value
    portfolio['Portfolio Value'] = portfolio.sum(axis=1)
    return portfolio

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    portfolio = backtest_strategy(
        tickers,
        folder_path="/Users/jaxsonfryer/Desktop/Quant/financial_data",
        start_date="2015-01-01",
        end_date="2023-01-01"
    )

    if not portfolio.empty and 'Portfolio Value' in portfolio.columns:
        # Plot results
        portfolio['Portfolio Value'].plot(title='Portfolio Backtest', figsize=(10, 6))
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()
    else:
        print("Portfolio data is empty or invalid. Cannot plot results.")




#comment