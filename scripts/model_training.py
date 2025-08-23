# This script trains the best performing model based on the analysis

import os
import numpy as np
import pandas as pd
import contextlib 
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from preprocessing import create_preprocessor
from model_utils import load_data, save_pipeline, save_tuning_results

def run_training():
    """
    Executes the complete model training and tuning pipeline.
    """
    print("=== Starting Model Training for K-Nearest Neighbors ===")

    # Load Data
    df = load_data('../data/raw/heart.csv')

    # Split Data
    X= df.drop('target', axis=1)
    y= df['target']
    X_train_val, X_final_test, y_train_val, y_final_test= train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training+validation and final testing sets.")

    # split training+validation data into seperate sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)  # 0.25 of 0.8 is 0.2

    # Save the unseen final test data for later evaluation
    final_test_df= X_final_test.copy()
    final_test_df['target']= y_final_test
    final_test_output_path= '../data/processed/final_test_data.csv'
    os.makedirs(os.path.dirname(final_test_output_path), exist_ok=True)
    final_test_df.to_csv(final_test_output_path, index=False)
    print(f"Final unseen test data saved to {final_test_output_path}")

    # Create full pipeline
    preprocessor= create_preprocessor()
    pipeline= Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier())
    ])

    # Define parameter grid for K-Nearest Neighbors
    param_grid= {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Run GridSearchCV
    search= GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    print("Starting hyperparameter tuning for K-Nearest Neighbors...")
    # Redirect stderr to suppress the joblib traceback
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        search.fit(X_train, y_train)

    print(f"Best cross-validation score: {search.best_score_:.4f} ")
    print("Best model and hyperparameter found:")
    print(search.best_params_)

    # Save the detailed tuning results
    results_filepath= '../reports/metrics/tuning_results.json'
    save_tuning_results(search.cv_results_, search.best_params_, results_filepath)

    # Save the best pipeline
    best_pipeline= search.best_estimator_
    save_pipeline(best_pipeline, '../models/final_pipeline_knn.pkl')

    print("\n=== Model Training Complete ===")

if __name__ == '__main__':
    run_training()