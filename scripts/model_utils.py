# This script contains helper functions for saving and loading models.

import os
import json
import joblib
import pandas as pd

def load_data(path):
    """
    Loads a CSV file from the given path.
    """
    print(f"Loading the data from {path}....")
    return pd.read_csv(path)

def save_pipeline(pipeline, path):
    """
    Saves a scikit-learn pipeline to the specified path.
    """
    dir_name= os.path.dirname(path)
    os.makedirs(dir_name, exist_ok= True)
    joblib.dump(pipeline, path)
    print(f"Pipeline saved successfully to {path}.")

def load_pipeline(path):
    """
    Loads a scikit-learn pipeline from a specified path.
    """
    print(f"Loading pipeline from {path}...")
    return joblib.load(path)
    
def save_tuning_results(results, best_params, filepath):
    """
    Saves GridSearchCV/RandomizedSearchCV results and best parameters to a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        results_df = pd.DataFrame(results).sort_values(by= 'rank_test_score')

        # Creating a Dictionary to save
        output_data= {
            'best parameters': best_params,
            'results': results_df.to_dict(orient='records')
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Tuning results saved to {filepath}")
    except Exception as e:
        print(f"Error saving tuning results: {e}")