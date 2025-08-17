# This script trains the best performing model based on the analysis

import pandas as pd
import joblib
import os
from sklearn.naive_bayes import GaussianNB

def train_and_save_model():
    """
    Loads the processed training data, trains the selected model,
    and saves the trained model artifact.
    """
    print("--- Starting Model Training ---")

    # 1. Loading Processed Training Data
    try:
        train_df = pd.read_csv('../data/processed/train.csv')
        print("Processed training data loaded successfully.")
    except FileNotFoundError:
        print("Error: '../data/processed/train.csv' not found.")
        print("Please ensure you have run the preprocessing.py script first.")
        return

    # 2. Separating Features (X) and Target (y)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']

    # 3. Initialize and Train the Best Model
    print("Initializing and training the Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Saving the Trained Model
    os.makedirs('../models', exist_ok=True)
    
    # Saving the model using joblib
    model_path = '../models/logistic_regression_model.pkl'
    joblib.dump(model, model_path)
    print(f"Trained model saved successfully to '{model_path}'")
    
    print("--- Model Training Script Finished ---")

if __name__ == "__main__":
    train_and_save_model()