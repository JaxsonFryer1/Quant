import pandas as pd
import yfinance as yf
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def compute_basic_ratios(ticker):
    """
    Compute basic financial ratios for a given ticker.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Calculated financial ratios.
    """
    ratios = {}
    try:
        # Fetch financial data from yfinance
        stock = yf.Ticker(ticker)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        history = stock.history(period="1y")

        # Ensure data is not empty
        if financials.empty or balance_sheet.empty or cashflow.empty or history.empty:
            return None

        # Use the latest data
        latest_financials = financials.iloc[:, 0]
        latest_balance_sheet = balance_sheet.iloc[:, 0]
        latest_cashflow = cashflow.iloc[:, 0]

        # Calculate basic ratios
        net_income = latest_financials.get('Net Income From Continuing Operations')
        revenue = latest_financials.get('Total Revenue')
        total_assets = latest_balance_sheet.get('Total Assets')
        total_liabilities = latest_balance_sheet.get('Total Liab')
        equity = latest_balance_sheet.get('Total Stockholder Equity')
        current_assets = latest_balance_sheet.get('Total Current Assets')
        current_liabilities = latest_balance_sheet.get('Total Current Liabilities')
        operating_cash_flow = latest_cashflow.get('Total Cash From Operating Activities')

        # Check if necessary data is available
        if any(v is None for v in [net_income, revenue, total_assets, total_liabilities, equity, current_assets, current_liabilities, operating_cash_flow]):
            return None

        # Profitability Ratios
        ratios['Net Profit Margin'] = net_income / revenue if revenue != 0 else None
        ratios['Return on Assets (ROA)'] = net_income / total_assets if total_assets != 0 else None
        ratios['Return on Equity (ROE)'] = net_income / equity if equity != 0 else None

        # Liquidity Ratios
        ratios['Current Ratio'] = current_assets / current_liabilities if current_liabilities != 0 else None

        # Leverage Ratios
        ratios['Debt-to-Equity Ratio'] = total_liabilities / equity if equity != 0 else None

        # Valuation Ratios
        market_price = history['Close'].iloc[-1]
        shares_outstanding = stock.info.get('sharesOutstanding')

        if shares_outstanding and shares_outstanding != 0:
            eps = net_income / shares_outstanding
            ratios['Earnings Per Share (EPS)'] = eps
            ratios['Price-to-Earnings (P/E) Ratio'] = market_price / eps if eps != 0 else None
        else:
            ratios['Earnings Per Share (EPS)'] = None
            ratios['Price-to-Earnings (P/E) Ratio'] = None

        return {'Ticker': ticker, **ratios}
    except Exception as e:
        print(f"Error computing ratios for {ticker}: {e}")
        return None

def screen_stocks_parallel(tickers, criteria):
    """
    Screen stocks based on basic financial ratios using parallel processing.

    Args:
        tickers (list): List of stock tickers to screen.
        criteria (dict): Dictionary of criteria to filter stocks.

    Returns:
        DataFrame: Stocks that meet the criteria.
    """
    results = []

    def analyze_ticker(ticker):
        ratios = compute_basic_ratios(ticker)
        if ratios is None:
            return None
            

        # Check if ratios meet the criteria
        for key, (min_val, max_val) in criteria.items():
            value = ratios.get(key)
            if value is None or (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                return None
        return ratios

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, ticker): ticker for ticker in tickers}

        for future in as_completed(future_to_ticker):
            result = future.result()
            if result:
                results.append(result)

    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    # Read tickers from the output of Code 1
    tickers_file = "NYSE_tickers.csv"
    if os.path.exists(tickers_file):
        tickers_df = pd.read_csv(tickers_file)
        tickers = tickers_df['Ticker'].tolist()
    else:
        print(f"Tickers file {tickers_file} not found.")
        tickers = []

    if not tickers:
        print("No tickers to analyze.")
    else:
        # Define criteria for screening
        criteria = {
            'Net Profit Margin': (0.05, None),       # At least 5%
            'Return on Assets (ROA)': (0.05, None),  # At least 5%
            'Return on Equity (ROE)': (0.10, None),  # At least 10%
            'Current Ratio': (1.0, None),            # At least 1.0
            'Debt-to-Equity Ratio': (None, 3.0),     # No more than 3.0
            'Price-to-Earnings (P/E) Ratio': (None, 20)  # No more than 20
        }

        # Screen stocks based on basic ratios in parallel
        results_df = screen_stocks_parallel(tickers, criteria)

        # Save results to a CSV file
        output_file = "tickers_for_deeper_analysis.csv"
        if not results_df.empty:
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
        else:
            print("\nNo tickers met the criteria.")
