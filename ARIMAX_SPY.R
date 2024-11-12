# Load required libraries
library(quantmod)
library(ggplot2)
library(dplyr)
library(tidyr)
library(rugarch)
library(PerformanceAnalytics)
library(TTR)
library(lubridate)
library(forecast)

get_projected_date <- function(start_date, trading_days) {
  projected_date <- start_date
  while (trading_days > 0) {
    projected_date <- projected_date + 1
    # Skip weekends (Saturday = 6, Sunday = 7)
    if (weekdays(projected_date) %in% c("Saturday", "Sunday")) next
    trading_days <- trading_days - 1
  }
  return(projected_date)
}



# Step 1: Fetch S&P 500 and Futures Data
symbol <- "SPY"
getSymbols(symbol, src = "yahoo", from = "2022-01-01", to = Sys.Date())
sp500_data <- na.omit(Cl(get(symbol)))
# Calculate dividend yield of SPY
dividends <- getDividends(symbol, from = "2022-01-01", to = Sys.Date())
annual_dividend <- sum(dividends) * (252 / length(index(dividends)))
dividend_yield <- annual_dividend / as.numeric(last(sp500_data))

# Futures Pricing
T_days <- 1
# Calculate projected close date
start_date <- Sys.Date()
projected_close_date <- get_projected_date(start_date, T_days)
T <- T_days / 252
getSymbols("DGS1MO", src = "FRED")
r <- as.numeric(last(na.omit(DGS1MO))) / 100
futures_price <- as.numeric(last(sp500_data)) * exp((r - dividend_yield) * T)

# Step 2: Fetch Macroeconomic Indicators
getSymbols("FEDFUNDS", src = "FRED", from = "2022-01-01")
fed_funds <- na.omit(FEDFUNDS)
getSymbols("CPIAUCSL", src = "FRED", from = "2022-01-01")
cpi_data <- na.omit(CPIAUCSL)
inflation_rate <- diff(log(cpi_data)) * 100
getSymbols("PAYEMS", src = "FRED", from = "2022-01-01")
employment_data <- na.omit(PAYEMS)

# Merge Data
macro_data <- Reduce(function(x, y) merge(x, y, join = "inner"), list(fed_funds, inflation_rate, employment_data))
colnames(macro_data) <- c("FedFundsRate", "InflationRate", "Employment")
macro_data_monthly <- to.monthly(macro_data, indexAt = "lastof", OHLC = FALSE)

# Step 3: S&P 500 Returns
sp500_monthly <- to.monthly(sp500_data, indexAt = "lastof", OHLC = FALSE)
sp500_returns <- monthlyReturn(sp500_monthly, type = "log")
data_combined <- merge.xts(sp500_returns, macro_data_monthly, join = "inner")
colnames(data_combined)[1] <- "SP500_Returns"
data_combined <- na.omit(data_combined)

# Step 4: ARIMAX Model
data_df <- data.frame(date = index(data_combined), coredata(data_combined))
data_df_clean <- na.omit(data_df)
xreg_matrix <- as.matrix(data_df_clean[, c("FedFundsRate", "InflationRate", "Employment")])
arimax_model <- auto.arima(data_df_clean$SP500_Returns, xreg = xreg_matrix, seasonal = FALSE)
forecast_result <- forecast(arimax_model, xreg = tail(xreg_matrix, 1), h = 1)
expected_return <- as.numeric(forecast_result$mean)

# Step 5: Adjust Drift
expected_return_annualized <- expected_return * 12
adjusted_drift <- max(r, expected_return_annualized)

# Step 6: Estimate Volatility
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
                   distribution.model = "std")
garch_fit <- ugarchfit(spec = spec, data = sp500_returns)

# Check for convergence and extract volatility
if (garch_fit@fit$convergence == 0) {
  volatility <- as.numeric(sigma(garch_fit))[length(sp500_returns)]
} else {
  stop("GARCH model did not converge.")
}

# Ensure adjusted_drift is numeric and not NA
if (!is.na(adjusted_drift) && length(adjusted_drift) == 1) {
  adjusted_drift <- as.numeric(adjusted_drift)
} else {
  stop("Adjusted drift is not properly calculated.")
}

# Step 7: Simulate Price Paths
simulate_paths <- function(S0, drift, T, n_steps, n_scenarios, volatility) {
  dt <- T / n_steps
  diffusion <- volatility * sqrt(dt)
  Z <- matrix(rnorm(n_steps * n_scenarios), nrow = n_steps, ncol = n_scenarios)
  log_returns <- (drift - 0.5 * volatility^2) * dt + diffusion * Z
  cum_log_returns <- apply(log_returns, 2, cumsum)
  price_matrix <- S0 * exp(cum_log_returns)
  price_matrix <- rbind(rep(S0, n_scenarios), price_matrix)
  return(price_matrix)
}

S0 <- as.numeric(last(sp500_data))
n_scenarios <- 1000
price_paths <- simulate_paths(S0, adjusted_drift, T, T_days, n_scenarios, volatility)

# Step 8: Plot Probability Cone
percentiles <- c(0.05, 0.25, 0.5, 0.75, 0.95)
quantiles_matrix <- apply(price_paths, 1, quantile, probs = percentiles)
plot_data <- data.frame(
  Days = 0:T_days,
  Quantile_5pct = quantiles_matrix[1, ],
  Quantile_25pct = quantiles_matrix[2, ],
  Median = quantiles_matrix[3, ],
  Quantile_75pct = quantiles_matrix[4, ],
  Quantile_95pct = quantiles_matrix[5, ]
)

ggplot(plot_data, aes(x = Days)) +
  geom_ribbon(aes(ymin = Quantile_5pct, ymax = Quantile_95pct), fill = "lightgreen", alpha = 0.5) +
  geom_ribbon(aes(ymin = Quantile_25pct, ymax = Quantile_75pct), fill = "green", alpha = 0.3) +
  geom_line(aes(y = Median), color = "darkgreen", size = 1) +
  labs(title = "S&P 500 1-Month Probability Cone with Macroeconomic Indicators",
       subtitle = "Incorporating Futures Pricing and Macroeconomic Data",
       x = "Days Ahead", y = "Price Level") +
  theme_minimal()

# Step 9: Summarize Key Statistics
final_prices <- price_paths[T_days + 1, ]
expected_price <- mean(final_prices)
median_price <- median(final_prices)
prob_up <- mean(final_prices > S0)
prob_down <- mean(final_prices < S0)
expected_return_pct <- (expected_price / S0 - 1) * 100

cat("Expected Price in ", T_days, " Trading Days (", format(projected_close_date, "%Y-%m-%d"), ") close: $", round(expected_price, 2), "\n")

cat("Expected Return: ", round(expected_return_pct, 2), "%\n")
cat("Probability of Price Increase: ", round(prob_up * 100, 2), "%\n")
cat("Probability of Price Decrease: ", round(prob_down * 100, 2), "%\n")
