# This script trains the best performing model based on the analysis

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from model_utils import save_model, save_tuning_results

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

    # 3. Initialize Model and define Hyperparameter grid
    print("Initializing Logistic Regression model for tuning...")
    model = LogisticRegression(max_iter=2000, random_state=42)
    
    # Define the hyperparameter grid to search
    param_grid ={
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    # Setup and run GridSearchCV
    print("Starting hyperparameter tuning with GridSeachCV...")
    grid_search= GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Display and log best results
    print(f"\nBest Parameters found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    # 4. Saving the Best Trained Model
    best_model= grid_search.best_estimator_
    model_path= '../models/final_model.pkl'
    save_model(best_model, model_path)

    # Save  the tuning results for analysis
    results_path= 'reports/metrics/tuning_results.json'
    save_tuning_results(grid_search.cv_results_, grid_search.best_params_, results_path)
    
    print("--- Model Training Script Finished ---")

if __name__ == "__main__":
    train_and_save_model()