import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.optimize import minimize

def tactical_weights(returns, momentum_window=60):
    """
    Calculates momentum-based tactical weights.
    
    Parameters:
    - returns: DataFrame of asset returns.
    - momentum_window: Lookback window for calculating momentum.
    
    Returns:
    - Series of normalized momentum weights.
    """
    # Compute rolling mean returns
    momentum = returns.rolling(window=int(momentum_window)).mean()
    # Use the last available momentum values
    last_momentum = momentum.iloc[-1]
    # Normalize weights to sum to 1
    momentum_weights = last_momentum / last_momentum.sum()
    return momentum_weights

def risk_parity_weights(cov_matrix):
    """
    Calculates risk parity weights based on inverse volatility.
    
    Parameters:
    - cov_matrix: Covariance matrix of asset returns.
    
    Returns:
    - Series of risk parity weights.
    """
    inv_volatility = 1 / np.sqrt(np.diag(cov_matrix))
    weights = inv_volatility / np.sum(inv_volatility)
    return pd.Series(weights, index=cov_matrix.index)

def skewness_hedge_signal(returns, threshold=-1.5):
    """
    Generates a hedge signal based on rolling skewness.
    
    Parameters:
    - returns: Series of asset returns.
    - threshold: Skewness threshold for generating hedge signals.
    
    Returns:
    - Series of hedge signals (1 for hedge, 0 for no hedge).
    """
    rolling_skew = returns.rolling(window=30).apply(skew, raw=True)
    hedge_signal = (rolling_skew < threshold).astype(int)
    return hedge_signal

def load_data(tickers, start_date, end_date):
    """
    Loads adjusted closing prices and computes returns.
    
    Parameters:
    - tickers: List of ticker symbols.
    - start_date: Start date for historical data.
    - end_date: End date for historical data.
    
    Returns:
    - Tuple of price DataFrame and returns DataFrame.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns

def backtest(returns, weights, hedge_signal):
    """
    Runs backtest of the strategy.
    
    Parameters:
    - returns: DataFrame of asset returns.
    - weights: Series of asset weights.
    - hedge_signal: Series of hedge signals.
    
    Returns:
    - Tuple of cumulative returns Series and Sharpe ratio.
    """
    # Compute portfolio returns
    portfolio_returns = returns.dot(weights)
    # Apply hedge signal
    hedged_returns = portfolio_returns * (1 - hedge_signal.reindex(portfolio_returns.index).fillna(0))
    # Compute cumulative returns
    cumulative_returns = (1 + hedged_returns).cumprod()
    # Compute Sharpe ratio
    sharpe_ratio = hedged_returns.mean() / hedged_returns.std() * np.sqrt(252)
    return cumulative_returns, sharpe_ratio

def sharpe_ratio_objective(params, returns):
    """
    Objective function to maximize the Sharpe ratio.
    
    Parameters:
    - params: Array of strategy parameters [alpha, skew_threshold, momentum_window].
    - returns: DataFrame of asset returns.
    
    Returns:
    - Negative Sharpe ratio (to be minimized).
    """
    alpha, skew_threshold, momentum_window = params

    # Validate momentum window
    if momentum_window < 1:
        return np.inf  # Invalid momentum window

    # Calculate weights
    cov_matrix = returns.cov()
    risk_weights = risk_parity_weights(cov_matrix)
    momentum_weights = tactical_weights(returns, momentum_window)
    hybrid_weights = alpha * risk_weights + (1 - alpha) * momentum_weights

    # Hedge signals
    hedge_signal = skewness_hedge_signal(returns['SPY'], threshold=skew_threshold)

    # Backtest
    cumulative_returns, sharpe_ratio = backtest(returns, hybrid_weights, hedge_signal)

    # Apply penalty for negative Sharpe ratio
    penalty = 0 if sharpe_ratio > 0 else abs(sharpe_ratio) * 10
    # penalty = 0
    return -(sharpe_ratio - penalty)  # Negative for minimization

def optimize_strategy(returns, sharpe_target=1.2, max_attempts=5):
    """
    Optimize strategy parameters to maximize Sharpe ratio.
    
    Parameters:
    - returns: DataFrame of asset returns.
    - sharpe_target: Target Sharpe ratio.
    - max_attempts: Maximum attempts to meet Sharpe target.
    
    Returns:
    - Tuple of optimal parameters and Sharpe ratio.
    """
    initial_params = [0.5, -1.5, 58]  # [alpha, skew_threshold, momentum_window]
    bounds = [(0, 2), (-3, 0), (20, 120)]  # Parameter bounds
    best_sharpe_ratio = float('-inf')
    best_params = None

    for attempt in range(max_attempts):
        result = minimize(
            sharpe_ratio_objective,
            initial_params,
            args=(returns,),
            bounds=bounds,
            method='L-BFGS-B'
        )
        optimal_params = result.x
        current_sharpe_ratio = -result.fun  # Negate because we minimized the negative Sharpe

        if current_sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = current_sharpe_ratio
            best_params = optimal_params

        # If target Sharpe ratio is met, return immediately
        if current_sharpe_ratio >= sharpe_target:
            print(f"Sharpe ratio target achieved on attempt {attempt + 1}: {current_sharpe_ratio:.2f}")
            return best_params, best_sharpe_ratio

        # Randomize initial parameters for next attempt
        initial_params = np.random.uniform([0, -3, 20], [1, 0, 120])

    # Output best Sharpe ratio attained if target not met
    print(f"Maximum Sharpe ratio attained: {best_sharpe_ratio:.2f} (Target: {sharpe_target})")
    return best_params, best_sharpe_ratio

if __name__ == "__main__":
    tickers = ['SPY', 'BITO', 'HYG', 'DBC']  # Replace 'TLT' with 'HYG'
    start_date = "2015-01-01"
    end_date = "2024-10-01"

    # Load data
    data, returns = load_data(tickers, start_date, end_date)

    # Optimize strategy
    optimal_params, max_sharpe_ratio = optimize_strategy(returns, sharpe_target=1.0, max_attempts=5)

    # Extract optimal parameters
    alpha, skew_threshold, momentum_window = optimal_params

    # Final weights and signals
    cov_matrix = returns.cov()
    risk_weights = risk_parity_weights(cov_matrix)
    momentum_weights = tactical_weights(returns, momentum_window)
    hybrid_weights = alpha * risk_weights + (1 - alpha) * momentum_weights
    hedge_signal = skewness_hedge_signal(returns['SPY'], threshold=skew_threshold)

    # Final backtest
    cumulative_returns, sharpe_ratio = backtest(returns, hybrid_weights, hedge_signal)

    # S&P 500 cumulative returns for comparison
    sp500_cumulative = (1 + returns['SPY']).cumprod()

    # Plot performance
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label="Strategy Cumulative Returns", linewidth=2)
    plt.plot(sp500_cumulative, label="S&P 500 Cumulative Returns", linestyle="--", linewidth=2, color='orange')
    plt.title(f"Optimized Strategy vs. S&P 500 | Sharpe Ratio: {sharpe_ratio:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.show()

    print("Optimal Parameters:")
    print(f"  Alpha (Risk/Momentum Blend): {alpha:.2f}")
    print(f"  Skew Threshold: {skew_threshold:.2f}")
    print(f"  Momentum Window: {int(momentum_window)}")
    print(f"Max Sharpe Ratio: {max_sharpe_ratio:.2f}")
