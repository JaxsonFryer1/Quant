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
library(scales)
library(tscv)

# Step 1: Fetch S&P 500 and Futures Data
fetch_data <- function(symbol, start_date, end_date) {
  getSymbols(symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = TRUE)
  return(Cl(get(symbol)))
}

sp500_data <- fetch_data("SPY", "2022-01-01", Sys.Date())

# Dividend Yield Calculation
calculate_dividend_yield <- function(symbol, sp500_data) {
  dividends <- getDividends(symbol, from = "2022-01-01", to = Sys.Date())
  annual_dividend <- sum(dividends) * (252 / length(index(dividends)))
  return(annual_dividend / as.numeric(last(sp500_data)))
}

dividend_yield <- calculate_dividend_yield("SPY", sp500_data)

# Fetch Risk-Free Rate
getSymbols("DGS1MO", src = "FRED")
r <- as.numeric(last(na.omit(DGS1MO))) / 100

# Theoretical Futures Price Calculation
calculate_futures_price <- function(spot_price, r, q, T) {
  return(spot_price * exp((r - q) * T))
}

T_days <- 21  # 1 month
T <- T_days / 252
futures_price <- calculate_futures_price(as.numeric(last(sp500_data)), r, dividend_yield, T)

# Step 2: Fetch Macroeconomic Indicators
fetch_macro_data <- function() {
  getSymbols(c("FEDFUNDS", "CPIAUCSL", "PAYEMS", "^VIX"), src = "FRED", from = "2022-01-01")
  fed_funds <- na.omit(FEDFUNDS)
  cpi_data <- na.omit(CPIAUCSL)
  employment_data <- na.omit(PAYEMS)
  vix_data <- na.omit(Cl(VIX))
  inflation_rate <- diff(log(cpi_data)) * 100  # Monthly inflation rate
  return(merge.xts(fed_funds, inflation_rate, employment_data, vix_data, join = "inner"))
}

macro_data <- fetch_macro_data()
colnames(macro_data) <- c("FedFundsRate", "InflationRate", "Employment", "VIX")
macro_data_monthly <- to.monthly(macro_data, indexAt = "lastof", OHLC = FALSE)

# Step 3: Prepare S&P 500 Returns Data
prepare_returns_data <- function(sp500_data, macro_data_monthly) {
  sp500_monthly <- to.monthly(sp500_data, indexAt = "lastof", OHLC = FALSE)
  sp500_returns <- monthlyReturn(sp500_monthly, type = "log")
  data_combined <- merge.xts(sp500_returns, macro_data_monthly, join = "inner")
  colnames(data_combined)[1] <- "SP500_Returns"
  return(na.omit(data_combined))
}

data_combined <- prepare_returns_data(sp500_data, macro_data_monthly)

# Step 4: Build ARIMAX Model with Lags
fit_arimax_model <- function(data_xts) {
  # Convert xts object to data frame for dplyr operations
  data_df <- as.data.frame(data_xts)
  data_df$Date <- index(data_xts)  # Preserve dates for reference
  
  # Ensure column names are valid
  colnames(data_df) <- make.names(colnames(data_df))
  
  # Differencing macroeconomic variables to ensure stationarity and adding lags
  diffed_data <- data_df %>%
    mutate(
      FedFundsRate = c(NA, diff(FedFundsRate)),
      InflationRate = c(NA, diff(InflationRate)),
      Employment = c(NA, diff(Employment)),
      VIX = c(NA, diff(VIX)),
      Lagged_SP500_Returns = lag(SP500_Returns, k = 1),
      Lagged_FedFundsRate = lag(FedFundsRate, k = 1),
      Lagged_InflationRate = lag(InflationRate, k = 1),
      Lagged_Employment = lag(Employment, k = 1),
      Lagged_VIX = lag(VIX, k = 1)
    ) %>%
    drop_na()  # Remove rows with NAs resulting from differencing
  
  # Scale the exogenous variables
  xreg_matrix <- scale(as.matrix(diffed_data[, c("FedFundsRate", "InflationRate", "Employment", "VIX",
                                                 "Lagged_SP500_Returns", "Lagged_FedFundsRate",
                                                 "Lagged_InflationRate", "Lagged_Employment", "Lagged_VIX")]))
  
  # Fit ARIMAX model using SP500_Returns and exogenous variables
  arimax_model <- auto.arima(
    diffed_data$SP500_Returns,
    xreg = xreg_matrix,
    seasonal = FALSE,
    stepwise = FALSE,
    approximation = FALSE
  )
  
  return(list(model = arimax_model, diffed_data = diffed_data))
}

# Apply the function to fit the ARIMAX model
arimax_results <- fit_arimax_model(data_combined)
arimax_model <- arimax_results$model
diffed_data <- arimax_results$diffed_data

# Check residuals for diagnostics
checkresiduals(arimax_model)

# Step 5: Use GARCH for Volatility Estimation
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "std"
)

garch_fit <- ugarchfit(spec = garch_spec, data = diffed_data$SP500_Returns)

# Extract conditional volatility
garch_volatility <- as.numeric(sigma(garch_fit)[length(sigma(garch_fit))])

# Step 6: Simulate Price Paths
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
price_paths <- simulate_paths(S0, r, T, T_days, n_scenarios, garch_volatility)

# Step 7: Visualize Probability Cone
plot_probability_cone <- function(price_paths, T_days) {
  percentiles <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  quantiles_matrix <- apply(price_paths, 1, quantile, probs = percentiles)
  time_vector <- 0:T_days
  plot_data <- data.frame(
    Days = time_vector,
    Quantile_5pct = quantiles_matrix[1, ],
    Quantile_25pct = quantiles_matrix[2, ],
    Median = quantiles_matrix[3, ],
    Quantile_75pct = quantiles_matrix[4, ],
    Quantile_95pct = quantiles_matrix[5, ]
  )
  ggplot(plot_data, aes(x = Days)) +
    geom_ribbon(aes(ymin = Quantile_5pct, ymax = Quantile_95pct), fill = "lightblue", alpha = 0.5) +
    geom_ribbon(aes(ymin = Quantile_25pct, ymax = Quantile_75pct), fill = "blue", alpha = 0.3) +
    geom_line(aes(y = Median), color = "darkblue", size = 1) +
    labs(
      title = "S&P 500 1-Month Probability Cone with Risk Zones",
      subtitle = "Highlighting Downturn Risks",
      x = "Days Ahead",
      y = "Price Level"
    ) +
    theme_minimal()
}

plot_probability_cone(price_paths, T_days)

# Step 8: Summarize Results
summarize_results <- function(price_paths, S0) {
  final_prices <- price_paths[nrow(price_paths), ]
  expected_price <- mean(final_prices)
  median_price <- median(final_prices)
  prob_up <- mean(final_prices > S0)
  prob_down <- mean(final_prices < S0)
  expected_return_pct <- (expected_price / S0 - 1) * 100
  cat("Summary of Enhanced Simulation Results:\n")
  cat("Expected Price in 1 Month: $", round(expected_price, 2), "\n")
  cat("Median Price in 1 Month: $", round(median_price, 2), "\n")
  cat("Expected Return: ", round(expected_return_pct, 2), "%\n")
  cat("Probability of Price Increase: ", round(prob_up * 100, 2), "%\n")
  cat("Probability of Price Decrease: ", round(prob_down * 100, 2), "%\n")
}

summarize_results(price_paths, S0)
