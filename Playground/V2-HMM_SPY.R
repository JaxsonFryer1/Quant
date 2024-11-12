# Load required libraries
library(quantmod)
library(xts)
library(dplyr)
library(TTR)
library(lubridate)
library(caret)
library(ggplot2)
library(PerformanceAnalytics)
library(randomForest)
library(xgboost)
library(moments)  # For skewness and kurtosis

# Descriptive Statistics
calculate_descriptive_stats <- function(data) {
  summary_stats <- data.frame(
    Feature = colnames(data),
    Mean = sapply(data, mean, na.rm = TRUE),
    Median = sapply(data, median, na.rm = TRUE),
    StdDev = sapply(data, sd, na.rm = TRUE),
    Skewness = sapply(data, skewness, na.rm = TRUE),
    Kurtosis = sapply(data, kurtosis, na.rm = TRUE)
  )
  return(summary_stats)
}

# Apply descriptive statistics on features and target
features_only <- train_data %>%
  select(-Date, -Future_5Day_Returns, -Future_Direction)
stats <- calculate_descriptive_stats(features_only)
print("Descriptive Statistics for Features:")
print(stats)

# Correlation Matrix
correlation_matrix <- cor(features_only, use = "complete.obs")
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize Correlation Matrix

if (!requireNamespace("reshape2", quietly = TRUE)) {
  install.packages("reshape2")
}
library(reshape2)
correlation_data <- melt(correlation_matrix)

ggplot(correlation_data, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Matrix", x = "Features", y = "Features")
# Backtesting Module
backtest_model <- function(data, best_model_name, window_size = 252 * 2, prediction_horizon = 5) {
  # Initialize vectors to store results
  dates <- c()
  actual_direction <- c()
  predicted_direction <- c()
  
  for (i in (window_size + 1):(nrow(data) - prediction_horizon)) {
    # Training data
    train_data <- data[(i - window_size):(i - 1), ]
    X_train <- train_data %>%
      select(-Date, -Future_5Day_Returns, -Future_Direction) %>%
      as.matrix()
    y_train <- ifelse(train_data$Future_Direction == -1, 0, 1)
    
    # Testing data
    test_data <- data[i, ]
    X_test <- test_data %>%
      select(-Date, -Future_5Day_Returns, -Future_Direction) %>%
      as.matrix()
    y_test <- ifelse(test_data$Future_Direction == -1, 0, 1)
    
    # Standardize features
    preProcValues <- preProcess(X_train, method = c("center", "scale"))
    X_train_scaled <- predict(preProcValues, X_train)
    X_test_scaled <- predict(preProcValues, X_test)
    
    # Train the best model
    if (best_model_name == "Logistic Regression") {
      model <- glm(y_train ~ ., data = as.data.frame(X_train_scaled), family = binomial)
      prob <- predict(model, newdata = as.data.frame(X_test_scaled), type = "response")
      pred <- ifelse(prob > 0.5, 1, -1)
    } else if (best_model_name == "Random Forest") {
      model <- randomForest(x = X_train_scaled, y = as.factor(y_train), ntree = 100)
      pred <- predict(model, newdata = X_test_scaled)
      pred <- as.numeric(as.character(pred))
    } else if (best_model_name == "XGBoost") {
      model <- xgboost(
        data = X_train_scaled,
        label = y_train,
        nrounds = 100,
        objective = "binary:logistic",
        verbose = 0
      )
      prob <- predict(model, X_test_scaled)
      pred <- ifelse(prob > 0.5, 1, -1)
    }
    
    # Store results
    dates <- c(dates, test_data$Date)
    actual_direction <- c(actual_direction, ifelse(y_test == 1, 1, -1))
    predicted_direction <- c(predicted_direction, pred)
  }
  
  # Create results data frame
  results <- data.frame(
    Date = as.Date(dates),
    Actual_Direction = actual_direction,
    Predicted_Direction = predicted_direction
  )
  
  # Merge with actual returns
  results <- merge(results, data[, c("Date", "Future_5Day_Returns")], by = "Date", all.x = TRUE)
  
  # Calculate Performance Metrics
  confusion <- confusionMatrix(
    factor(results$Predicted_Direction, levels = c(-1, 1)),
    factor(results$Actual_Direction, levels = c(-1, 1))
  )
  
  accuracy <- confusion$overall['Accuracy']
  cat("\nBacktest Accuracy:", round(accuracy * 100, 2), "%\n")
  
  # Strategy Returns
  strategy_returns <- ifelse(results$Predicted_Direction == 1, results$Future_5Day_Returns,
                             ifelse(results$Predicted_Direction == -1, -results$Future_5Day_Returns, 0))
  
  # Include transaction costs
  transaction_cost <- 0.001  # 0.1% per trade
  trade_signals <- c(1, diff(results$Predicted_Direction) != 0)  # Trade when signal changes
  strategy_returns_after_cost <- strategy_returns - (transaction_cost * trade_signals)
  
  # Cumulative returns
  cumulative_returns <- cumprod(1 + strategy_returns_after_cost) - 1
  
  # Sharpe Ratio
  average_return <- mean(strategy_returns_after_cost) * (252 / 5)
  sd_return <- sd(strategy_returns_after_cost) * sqrt(252 / 5)
  sharpe_ratio <- average_return / sd_return
  
  # Maximum Drawdown
  drawdowns <- Drawdowns(strategy_returns_after_cost)
  max_drawdown <- max(drawdowns)
  
  # Output Performance Metrics
  cat("Confusion Matrix:\n")
  print(confusion$table)
  cat("\n")
  cat("Overall Accuracy:", round(accuracy * 100, 2), "%\n")
  cat("Annualized Sharpe Ratio:", round(sharpe_ratio, 2), "\n")
  cat("Maximum Drawdown:", round(max_drawdown * 100, 2), "%\n")
  
  # Plot cumulative returns
  ggplot(data.frame(Date = results$Date, Cumulative_Returns = cumulative_returns), aes(x = Date, y = Cumulative_Returns)) +
    geom_line(color = "blue") +
    labs(title = "Cumulative Returns of the Strategy", x = "Date", y = "Cumulative Returns") +
    theme_minimal()
}

# Run Backtest
backtest_model(data_df, best_model_name)
