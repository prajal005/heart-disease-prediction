
# model_training.py
# This script is used for training the ML model.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    # Load dataset
    data = pd.read_csv('data/dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
