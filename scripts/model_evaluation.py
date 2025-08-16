
# model_evaluation.py
# This script evaluates the trained ML model.
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate_model():
    # Load test dataset
    data = pd.read_csv('data/test_dataset.csv')
    X_test = data.drop('target', axis=1)
    y_test = data['target']

    # Load the trained model
    model = joblib.load('models/random_forest_model.pkl')

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
