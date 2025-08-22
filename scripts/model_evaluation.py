# This script evaluates the trained ML model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model_utils import load_data, load_pipeline

def run_evaluation():
    """
    Loads the saved pipeline and evaluates it into the test set.
    """
    print("=== Starting Model Evaluation ===")

    # Load Data and split it exactly like in training to get the test set
    df= load_data('../data/raw/heart.csv')
    X= df.drop('target', axis=1)
    y= df['target']
    _, X_test, _, y_test= train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load the trained pipeline
    pipeline= load_pipeline('../models/final_pipeline.pkl')

    # Prediction on the raw test data
    y_pred= pipeline.predict(X_test)

    # Print evaluation metrics
    accuracy= accuracy_score(y_test, y_pred)
    report= classification_report(y_test, y_pred)
    cm= confusion_matrix(y_test, y_pred)

    print(f"\nFinal Model Accuracy on Test Set: {accuracy:.4f}")
    print('\nClassification Report:')
    print(report)
    print('\nConfusion Matrix:')
    print(cm)

    print("\n=== Model Evaluation Complete ===")

if __name__ == '__main__':
    run_evaluation()