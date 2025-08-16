
# data_preprocessing.py
# This script handles data loading, cleaning, and preprocessing for ML.
import pandas as pd
import numpy as np

def load_data(path):
    # Load dataset from the specified path
    return pd.read_csv(path)

def clean_data(df):
    # Perform data cleaning
    return df.dropna()

if __name__ == "__main__":
    # Example usage
    data = load_data('data/dataset.csv')
    data = clean_data(data)
    print(data.head())
