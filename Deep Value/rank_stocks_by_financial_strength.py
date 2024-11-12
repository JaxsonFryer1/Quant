import pandas as pd
import yfinance as yf
import os

def compute_detailed_ratios(ticker):
    """
    Compute detailed financial ratios and metrics for a given ticker.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Calculated financial ratios and metrics.
    """
    ratios = {}
    try:
        # Fetch financial data from yfinance
        stock = yf.Ticker(ticker)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        history = stock.history(period="5y")

        # Ensure data is not empty
        if financials.empty or balance_sheet.empty or cashflow.empty or history.empty:
            return None

        # Use the latest data
        latest_financials = financials.iloc[:, 0]
        latest_balance_sheet = balance_sheet.iloc[:, 0]
        latest_cashflow = cashflow.iloc[:, 0]

        # Calculate detailed ratios
        net_income = latest_financials.get('Net Income')
        revenue = latest_financials.get('Total Revenue')
        gross_profit = latest_financials.get('Gross Profit')
        operating_income = latest_financials.get('Operating Income')
        total_assets = latest_balance_sheet.get('Total Assets')
        total_liabilities = latest_balance_sheet.get('Total Liab')
        equity = latest_balance_sheet.get('Total Stockholder Equity')
        current_assets = latest_balance_sheet.get('Total Current Assets')
        current_liabilities = latest_balance_sheet.get('Total Current Liabilities')
        inventory = latest_balance_sheet.get('Inventory')
        accounts_receivable = latest_balance_sheet.get('Net Receivables')
        operating_cash_flow = latest_cashflow.get('Total Cash From Operating Activities')
        capital_expenditures = latest_cashflow.get('Capital Expenditures')

        # Market data
        market_price = history['Close'].iloc[-1]
        shares_outstanding = stock.info.get('sharesOutstanding')
        beta = stock.info.get('beta')

        # Check if necessary data is available
        required_data = [net_income, revenue, gross_profit, operating_income, total_assets,
                         total_liabilities, equity, current_assets, current_liabilities,
                         operating_cash_flow, market_price, shares_outstanding]
        if any(v is None for v in required_data):
            return None

        # Profitability Ratios
        ratios['Gross Profit Margin'] = gross_profit / revenue if revenue != 0 else None
        ratios['Operating Margin'] = operating_income / revenue if revenue != 0 else None
        ratios['Net Profit Margin'] = net_income / revenue if revenue != 0 else None
        ratios['Return on Assets (ROA)'] = net_income / total_assets if total_assets != 0 else None
        ratios['Return on Equity (ROE)'] = net_income / equity if equity != 0 else None

        # Liquidity Ratios
        ratios['Current Ratio'] = current_assets / current_liabilities if current_liabilities != 0 else None
        ratios['Quick Ratio'] = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else None

        # Efficiency Ratios
        if inventory and inventory != 0:
            cost_of_revenue = latest_financials.get('Cost Of Revenue')
            ratios['Inventory Turnover'] = cost_of_revenue / inventory if cost_of_revenue else None
        else:
            ratios['Inventory Turnover'] = None
        if accounts_receivable and accounts_receivable != 0:
            ratios['Receivables Turnover'] = revenue / accounts_receivable
        else:
            ratios['Receivables Turnover'] = None

        # Leverage Ratios
        ratios['Debt-to-Equity Ratio'] = total_liabilities / equity if equity != 0 else None
        ratios['Debt-to-Assets Ratio'] = total_liabilities / total_assets if total_assets != 0 else None

        # Valuation Ratios
        eps = net_income / shares_outstanding if shares_outstanding != 0 else None
        book_value_per_share = equity / shares_outstanding if shares_outstanding != 0 else None

        ratios['Earnings Per Share (EPS)'] = eps
        ratios['Price-to-Earnings (P/E) Ratio'] = market_price / eps if eps != 0 else None
        ratios['Price-to-Book (P/B) Ratio'] = market_price / book_value_per_share if book_value_per_share != 0 else None

        # Growth Ratios
        # Assuming we have multiple periods, we can compute growth rates
        revenue_growth = None
        net_income_growth = None
        if revenue and revenue != 0 and financials.shape[1] > 1:
            previous_revenue = financials.iloc[:, 1].get('Total Revenue')
            if previous_revenue and previous_revenue != 0:
                revenue_growth = (revenue - previous_revenue) / previous_revenue
        if net_income and net_income != 0 and financials.shape[1] > 1:
            previous_net_income = financials.iloc[:, 1].get('Net Income')
            if previous_net_income and previous_net_income != 0:
                net_income_growth = (net_income - previous_net_income) / previous_net_income

        ratios['Revenue Growth'] = revenue_growth
        ratios['Net Income Growth'] = net_income_growth

        # Other Metrics
        ratios['Beta'] = beta

        return ratios
    except Exception as e:
        print(f"Error computing detailed ratios for {ticker}: {e}")
        return None

def rank_stocks(tickers):
    """
    Rank stocks based on their financial strength.

    Args:
        tickers (list): List of stock tickers to analyze.

    Returns:
        DataFrame: Ranked stocks with their scores.
    """
    results = []
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        ratios = compute_detailed_ratios(ticker)
        if ratios is None:
            continue

        # Scoring system (simple example)
        score = 0
        # High ROE
        roe = ratios.get('Return on Equity (ROE)')
        if roe and roe > 0.15:
            score += 2
        elif roe and roe > 0.10:
            score += 1

        # Low Debt-to-Equity Ratio
        debt_to_equity = ratios.get('Debt-to-Equity Ratio')
        if debt_to_equity and debt_to_equity < 1.0:
            score += 2
        elif debt_to_equity and debt_to_equity < 2.0:
            score += 1

        # Consistent Growth
        revenue_growth = ratios.get('Revenue Growth')
        net_income_growth = ratios.get('Net Income Growth')
        if revenue_growth and revenue_growth > 0.05:
            score += 1
        if net_income_growth and net_income_growth > 0.05:
            score += 1

        # Low P/E Ratio
        pe_ratio = ratios.get('Price-to-Earnings (P/E) Ratio')
        if pe_ratio and pe_ratio < 15:
            score += 2
        elif pe_ratio and pe_ratio < 20:
            score += 1

        # Add to results
        result = {'Ticker': ticker, 'Score': score}
        result.update(ratios)
        results.append(result)

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Score', ascending=False)

    return results_df

if __name__ == "__main__":
    # Read tickers from the output of Code 2
    tickers_file = "tickers_for_deeper_analysis.csv"
    if os.path.exists(tickers_file):
        tickers_df = pd.read_csv(tickers_file)
        tickers = tickers_df['Ticker'].tolist()
    else:
        print(f"Tickers file {tickers_file} not found.")
        tickers = []

    if not tickers:
        print("No tickers to analyze.")
    else:
        # Rank stocks based on detailed analysis
        results_df = rank_stocks(tickers)

        # Save results to a CSV file
        output_file = "final_tickers_suggestions.csv"
        if not results_df.empty:
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
        else:
            print("\nNo tickers met the criteria.")
