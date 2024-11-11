import requests
import os

# Set up your API keys here
ALPHA_VANTAGE_API_KEY = '2JSP6T0HLHWV830U'
FRED_API_KEY = '1c4daf05ce00f1dd2456c524c02a3e5d'



# Fetch Consumer Confidence Index from FRED
def get_consumer_confidence_index():
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=CSCICP03USM665S&api_key={FRED_API_KEY}&file_type=json'
    response = requests.get(url)
    data = response.json()
    
    try:
        cci = float(data['observations'][-1]['value'])
        return cci
    except KeyError:
        print("Error fetching Consumer Confidence Index data.")
        return 100  # Default neutral value

# Fetch Unemployment Rate from FRED
def get_unemployment_rate():
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={FRED_API_KEY}&file_type=json'
    response = requests.get(url)
    data = response.json()
    
    try:
        unemployment_rate = float(data['observations'][-1]['value'])
        return unemployment_rate
    except KeyError:
        print("Error fetching Unemployment Rate data.")
        return 4  # Default neutral value

# Fetch Inflation Rate (CPI) from FRED
def get_inflation_rate():
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={FRED_API_KEY}&file_type=json'
    response = requests.get(url)
    data = response.json()
    
    try:
        # Convert current and previous month CPI values to float
        cpi = float(data['observations'][-1]['value'])
        previous_cpi = float(data['observations'][-13]['value'])  # Previous year’s CPI
        inflation_rate = (cpi - previous_cpi) / previous_cpi * 100
        return inflation_rate
    except (KeyError, IndexError, ValueError) as e:
        print("Error fetching Inflation Rate data:", e)
        return 2  # Default neutral value


# Fetch Interest Rate from FRED
def get_interest_rate():
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={FRED_API_KEY}&file_type=json'
    response = requests.get(url)
    data = response.json()
    
    try:
        interest_rate = float(data['observations'][-1]['value'])
        return interest_rate
    except KeyError:
        print("Error fetching Interest Rate data.")
        return 2  # Default neutral value

# Fetch GDP Growth from FRED
def get_gdp_growth():
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key={FRED_API_KEY}&file_type=json'
    response = requests.get(url)
    data = response.json()
    
    try:
        gdp_current = float(data['observations'][-1]['value'])
        gdp_previous = float(data['observations'][-2]['value'])
        gdp_growth = (gdp_current - gdp_previous) / gdp_previous * 100
        return gdp_growth
    except KeyError:
        print("Error fetching GDP Growth data.")
        return 2  # Default neutral value

# Additional functions for other indicators can follow similar structures

# Function to calculate macroeconomic sentiment
def calculate_macro_sentiment():
    indicators = {
        "Consumer Confidence Index": get_consumer_confidence_index(),
        "Unemployment Rate": get_unemployment_rate(),
        "Inflation Rate": get_inflation_rate(),
        "Interest Rate": get_interest_rate(),
        "GDP Growth": get_gdp_growth(),
        # Add other indicator functions here
    }
    


#     "
#     Indicator	Baseline (Neutral Value)	Normalization Formula
# Consumer Confidence Index	100	(value - 100) / 100
# Unemployment Rate	4%	(4 - value) / 4
# Inflation Rate (CPI)	2%	(2 - value) / 2
# Interest Rate	2%	(2 - value) / 2
# GDP Growth Rate	3%	value / 3
#     "
    # Normalize scores based on reasonable benchmarks
    normalized_scores = {}
    for key, value in indicators.items():
        if key == "Consumer Confidence Index":
            normalized_scores[key] = (value - 100) / 100
        elif key == "Unemployment Rate":
            normalized_scores[key] = (4 - value) / 4
        elif key == "Inflation Rate":
            normalized_scores[key] = (2 - value) / 2
        elif key == "Interest Rate":
            normalized_scores[key] = (2 - value) / 2
        elif key == "GDP Growth":
            normalized_scores[key] = value / 3

    # Aggregate scores and interpret
    overall_sentiment = sum(normalized_scores.values()) / len(normalized_scores)
    print("Individual Indicator Scores:")
    for key, score in normalized_scores.items():
        print(f"{key}: {score:.2f}")

    print(f"\nOverall Macroeconomic Sentiment Score: {overall_sentiment:.2f}")

    if overall_sentiment > 0.2:
        interpretation = "Positive Economic Outlook"
    elif overall_sentiment < -0.2:
        interpretation = "Negative Economic Outlook"
    else:
        interpretation = "Neutral Economic Outlook"

    print(f"Interpretation: {interpretation}")

# Run the macroeconomic sentiment analysis
calculate_macro_sentiment()
