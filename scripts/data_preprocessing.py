# preprocessing.py
# This script is dedicated to preprocessing the raw heart disease data.
# It performs the following steps:
# 1. Loads the raw data.
# 2. Splits the data into training and testing sets.
# 3. Defines a preprocessing pipeline to scale numerical features and one-hot encode categorical features.
# 4. Fits the pipeline on the training data.
# 5. Saves the fitted preprocessor object for later use in prediction.
# 6. Transforms both training and testing data and saves them to the processed data folder.

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def run_preprocessing():
    """
    Executes the full data preprocessing pipeline.
    """
    print("--- Starting Data Preprocessing ---")

    # 1. Load Raw Data
    try:
        df = pd.read_csv('../data/raw/heart.csv')
        print("Raw data loaded successfully.")
    except FileNotFoundError:
        print("Error: '../data/raw/heart.csv' not found. Make sure the path is correct.")
        return

    # 2. Define Features and Target
    X = df.drop('target', axis=1)
    y = df['target']

    # Identify feature types for the preprocessor
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # 3. Split into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training and testing sets.")

    # 4. Create the Preprocessing Pipeline
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features),('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)],remainder='passthrough')

    # 5. Fitting the Preprocessor and Saving It
    preprocessor.fit(X_train)
    print("Preprocessor fitted on the training data.")

    # Creating the models directory if it doesn't exist and saving the preprocessor
    os.makedirs('../models', exist_ok=True)
    joblib.dump(preprocessor, '../models/preprocessor.pkl')
    print("Fitted preprocessor saved to '../models/preprocessor.pkl'")

    # 6. Transforming Data and Saving Processed Files
    feature_names = preprocessor.get_feature_names_out()

    # Transforming the training data
    X_train_processed = preprocessor.transform(X_train)
    train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_processed_df['target'] = y_train.reset_index(drop=True)

    # Transforming the testing data using the SAME fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)
    test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_processed_df['target'] = y_test.reset_index(drop=True)

    # Creatinh the processed data directory and saving the files
    os.makedirs('../data/processed', exist_ok=True)
    train_processed_df.to_csv('../data/processed/train.csv', index=False)
    test_processed_df.to_csv('../data/processed/test.csv', index=False)
    print("Processed train and test sets saved to '../data/processed/'")

    print("--- Data Preprocessing Complete ---")

if __name__ == "__main__":
    run_preprocessing()
