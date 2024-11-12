

import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    
    # Read CSV with custom column names, no header
    df = pd.read_csv(file_path, names=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    print("Creating datetime index...")
    df.index = pd.date_range(start='2018-01-01', periods=len(df), freq='H')

    print("Calculating returns and volatility...")
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=24).std()

    print("Calculating volume change...")
    df['Volume_Change'] = df['Volume'].pct_change()

    print("Dropping NaN values...")
    df.dropna(inplace=True)

    print(f"Data preprocessed. Shape: {df.shape}")
    
    return df
