# This script trains the best performing model based on the analysis

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from preprocessing import create_preprocessor
from model_utils import load_data, save_pipeline, save_tuning_results

def run_training():
    """
    Executes the complete model training and tuning pipeline.
    """
    print("=== Starting Model Training for Logistic Regression ===")

    # Load Data
    df = load_data('../data/raw/heart.csv')

    # Split Data
    X= df.drop('target', axis=1)
    y= df['target']
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training and testing sets.")

    # Removal of outliers from the training set
    print("Removing outliers from the training set.....")
    outliers= (X_train['ca'] <= 3) & (X_train['thalach'] <= 200)
    X_train= X_train[outliers]
    y_train= y_train.loc[X_train.index]
    print("=== Outliers removed ===")

    # Create full pipeline
    preprocessor= create_preprocessor()
    pipeline= Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42))
    ])

    # Define parameter grid for Logistic Regression
    param_grid= {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga']
    }

    # Run GridSearchCV
    search= GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    print("Starting hyperparameter tuning for Logistic Regression...")
    search.fit(X_train, y_train)

    print(f"Best cross-validation score: {search.best_score_:.4f} ")
    print("Best model and hyperparameter found:")
    print(search.best_params_)

    # Save the detailed tuning results
    results_filepath= '../reports/metrics/tuning_results.json'
    save_tuning_results(search.cv_results_, search.best_params_, results_filepath)

    # Save the best pipeline
    best_pipeline= search.best_estimator_
    save_pipeline(best_pipeline, '../models/final_pipeline.pkl')

    print("\n=== Model Training Complete ===")

if __name__ == '__main__':
    run_training()