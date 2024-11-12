import os
import pandas as pd
import yfinance as yf

def fetch_and_save_data(tickers, folder_path="financial_data"):
    """
    Fetch financial data and save it in an organized folder structure.

    Args:
        tickers (list): List of stock tickers.
        folder_path (str): Path to store the financial data.
    """
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_folder_path = os.path.join(script_dir, folder_path)

    if not os.path.exists(absolute_folder_path):
        os.makedirs(absolute_folder_path)

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        print(f"Fetching data for {ticker}...")

        # Fetch historical stock data
        history = stock.history(period='10y')
        history.to_csv(os.path.join(absolute_folder_path, f"{ticker}_history.csv"))

        # Fetch financial statements
        try:
            stock_balance_sheet = stock.balance_sheet
            stock_balance_sheet.to_csv(os.path.join(absolute_folder_path, f"{ticker}_balance_sheet.csv"))

            stock_financials = stock.financials
            stock_financials.to_csv(os.path.join(absolute_folder_path, f"{ticker}_income_statement.csv"))

            stock_cashflow = stock.cashflow
            stock_cashflow.to_csv(os.path.join(absolute_folder_path, f"{ticker}_cashflow.csv"))
        except Exception as e:
            print(f"Failed to fetch financial data for {ticker}: {e}")

# Example usage:
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    fetch_and_save_data(tickers)
