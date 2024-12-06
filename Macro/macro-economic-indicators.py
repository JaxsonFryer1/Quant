import requests
import numpy as np
import pandas as pd
from datetime import datetime

# API keys
FRED_API_KEY = '1c4daf05ce00f1dd2456c524c02a3e5d'

# Fetch historical data from FRED
def fetch_fred_data(series_id, start_date="2000-01-01"):
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json'
    response = requests.get(url)
    data = response.json()
    try:
        observations = data['observations']
        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[df['date'] >= start_date]  # Filter by start date
        return df.set_index('date')['value']
    except KeyError:
        print(f"Error fetching data for {series_id}")
        return pd.Series(dtype=float)

# Fetch all required indicators
def get_all_indicators():
    indicators = {
        "Consumer Confidence Index": fetch_fred_data("CSCICP03USM665S"),
        "Unemployment Rate": fetch_fred_data("UNRATE"),
        "Inflation Rate": fetch_fred_data("CPIAUCSL"),
        "Interest Rate": fetch_fred_data("FEDFUNDS"),
        "GDP Growth": fetch_fred_data("A191RL1Q225SBEA"),
    }
    return indicators

# Calculate z-scores
def calculate_z_scores(series):
    return (series - series.mean()) / series.std()

# Calculate weighted sentiment
def calculate_weighted_sentiment(normalized_scores, weights):
    weighted_scores = {k: normalized_scores[k] * weights[k] for k in normalized_scores}
    return sum(weighted_scores.values()) / sum(weights.values())

# Main analysis function
def calculate_macro_sentiment():
    indicators = get_all_indicators()
    weights = {
        "Consumer Confidence Index": 0.25,
        "Unemployment Rate": 0.15,
        "Inflation Rate": 0.2,
        "Interest Rate": 0.25,
        "GDP Growth": 0.15,
    }
    normalized_scores = {}

    for key, series in indicators.items():
        z_scores = calculate_z_scores(series)
        normalized_scores[key] = z_scores[-1]  # Use the latest z-score for sentiment
    
    sentiment_score = calculate_weighted_sentiment(normalized_scores, weights)

    print("Individual Indicator Z-Scores:")
    for key, score in normalized_scores.items():
        print(f"{key}: {score:.2f}")

    print(f"\nOverall Macroeconomic Sentiment Score: {sentiment_score:.2f}")
    if sentiment_score > 0.5:
        interpretation = "Positive Economic Outlook"
    elif sentiment_score < -0.5:
        interpretation = "Negative Economic Outlook"
    else:
        interpretation = "Neutral Economic Outlook"
    
    print(f"Interpretation: {interpretation}")

# Run the macroeconomic sentiment analysis
calculate_macro_sentiment()
