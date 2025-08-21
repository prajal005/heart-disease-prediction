# This is a tkinter app for heart-disease-prediction

import tkinter as tk
import pandas as pd
import joblib
import os

# Setting up the path
app_dir= os.path.dirname(os.path.abspath(__file__))
project_dir= os.path.dirname(app_dir)
model_path= os.path.join(project_dir, "models", "final_model.pkl")
preprocessor_path= os.path.join(project_dir, "models", "preprocessor.pkl")

# Loading the  artifacts
