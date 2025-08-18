
# model_evaluation.py
# This script evaluates the trained ML model.
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model():
    """
    Loads the preprocessed test data, trained model and makes prediction.
    """
    print("--- Starting Model Evaluation ---")

    processed_test_data = '../data/processed/test.csv' 
    model_path = '../models/final_model.pkl' 

    # 1. Load Preprocessed Test Data
    try:
        test_df = pd.read_csv(processed_test_data)
        print(f"Preprocessed test data loaded successfully from '{processed_test_data}'")
    except FileNotFoundError:
        print(f"Error: '{processed_test_data}' not found. Please run the preprocessing script first.")
        return

    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # 2. Load Trained Model
    try:
        model = joblib.load(model_path)
        print(f"Trained model loaded successfully from '{model_path}'")
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{model_path}'. Please run the training script first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Make Predictions
    print("Making predictions on the test data...")
    y_pred = model.predict(X_test)
    print("Predictions completed.")

    # 4. Evaluate the Model
    print("\n--- Model Evaluation Results ---")
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(matrix)

    print("\n--- Model Evaluation Complete ---")

if __name__ == "__main__":
    evaluate_model()
