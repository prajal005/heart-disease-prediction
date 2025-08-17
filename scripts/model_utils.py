# This script contains helper functions for saving and loading models.

import joblib
import os

def save_model(model, filepath):
    """
    Saves a trained model to a file.

    Args:
        model: The trained machine learning model object.
        filepath (str): The path where the model will be saved (e.g., 'models/heart_disease_model.pkl').
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model to {filepath}: {e}")

def load_model(filepath):
    """
    Loads a trained model from a file.

    Args:
        filepath (str): The path to the saved model file.

    Returns:
        The loaded model object, or None if an error occurs.
    """
    try:
        model = joblib.load(filepath)
        print(f"Model loaded successfully from {filepath}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None