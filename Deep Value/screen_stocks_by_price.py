import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def read_nyse_tickers(file_path="NYSE_tickers.csv"):
    """
    Read NYSE tickers from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing NYSE tickers.

    Returns:
        list: List of stock tickers.
    """
    try:
        tickers_df = pd.read_csv(file_path)
        return tickers_df['Symbol'].tolist()
    except Exception as e:
        print(f"Error reading tickers file: {e}")
        return []

def analyze_ticker(ticker, threshold):
    """
    Analyze a single stock ticker for the given criteria.

    Args:
        ticker (str): Stock ticker symbol.
        threshold (float): Percentage off the high value.

    Returns:
        dict: Ticker analysis results or None if the ticker does not meet criteria.
    """
    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(period="max")

        # Skip if insufficient data
        if historical_data.empty:
            return None

        # Calculate metrics
        high_price = historical_data['High'].max()
        current_price = historical_data['Close'].iloc[-1]
        off_high_percentage = (high_price - current_price) / high_price

        # Check if stock is at least `threshold` off its high
        if off_high_percentage >= threshold:
            return {
                'Ticker': ticker,
                'High Price': high_price,
                'Current Price': current_price,
                '% Off High': off_high_percentage * 100
            }
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
    return None

def screen_stocks_below_high(tickers, threshold=0.5, max_stocks=None):
    """
    Screen for stocks at least `threshold` below their high over the last `years`.

    Args:
        tickers (list): List of stock tickers to screen.
        threshold (float): Percentage off the high value (e.g., 0.5 for 50% off).
        max_stocks (int): Maximum number of stocks to return (optional).

    Returns:
        DataFrame: Stocks that meet the criteria.
    """
    results = []

    # Use ThreadPoolExecutor to parallelize the analysis
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_ticker, ticker, threshold): ticker for ticker in tickers}

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Convert results to a DataFrame and sort
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='% Off High', ascending=False)

    if max_stocks:
        results_df = results_df.head(max_stocks)

    return results_df

if __name__ == "__main__":
    # Read tickers from the file
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NYSE_tickers.csv")
    tickers = read_nyse_tickers(file_path)

    if not tickers:
        print("No tickers found to analyze.")
    else:
        # Screen for stocks at least 50% below their high
        results_df = screen_stocks_below_high(tickers, threshold=0.5)

        # Save results to a CSV file
        output_file = "tickers_to_research.csv"
        if not results_df.empty:
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
        else:
            print("\nNo tickers met the criteria.")
