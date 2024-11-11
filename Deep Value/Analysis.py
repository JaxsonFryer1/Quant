import pandas as pd
import os
def compute_ratios(folder_path="financial_data", tickers=None):
    """
    Compute financial ratios for stored financial data.

    Args:
        folder_path (str): Path to stored financial data.
        tickers (list): List of stock tickers to analyze.

    Returns:
        dict: Financial ratios for each stock.
    """
    ratios = {}
    for ticker in tickers:
        try:
            print(f"Analyzing {ticker}...")
            # Load data
            income_statement = pd.read_csv(os.path.join(folder_path, f"{ticker}_income_statement.csv"), index_col=0)
            balance_sheet = pd.read_csv(os.path.join(folder_path, f"{ticker}_balance_sheet.csv"), index_col=0)

            # Compute metrics
            net_income = income_statement.loc['Net Income'].mean()
            revenue = income_statement.loc['Total Revenue'].mean()
            equity = balance_sheet.loc['Total Stockholder Equity'].mean()
            liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].mean()

            # Ratios
            roe = net_income / equity
            debt_to_equity = liabilities / equity
            profit_margin = net_income / revenue

            ratios[ticker] = {
                'ROE': roe,
                'Debt-to-Equity': debt_to_equity,
                'Profit Margin': profit_margin
            }
        except Exception as e:
            print(f"Failed to analyze {ticker}: {e}")
    
    return ratios

# Example usage:
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    ratios = compute_ratios(tickers=tickers)
    print("Financial Ratios:")
    print(ratios)
