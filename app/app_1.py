# This is a tkinter app for heart-disease-prediction

import os
import joblib
import pandas as pd
import tkinter as tk
from tkinter import messagebox

from tkinter import messagebox, ttk

# Setting up the path
try:
    app_dir= os.path.dirname(os.path.abspath(__file__))
    project_dir= os.path.dirname(app_dir)
    model_path= os.path.join(project_dir, "models", "final_pipeline_knn.pkl")

    # Loading the  artifacts
    pipeline= joblib.load(model_path)
    model_loaded= True
except FileNotFoundError as e:
    messagebox.showerror("Error loading model artifacts: {e}")
    print("Please make sure the 'final_pipeline_knn.pkl' file is in the 'models' directory.")
    pipeline= None
    model_loaded= False
except Exception as e:
    messagebox.showerror("Error",f"An unexpected error occured during the model loading: {e}")
    pipeline= None
    model_loaded= False

# -- APP LAYOUT --
class HeartDiseaseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Size of the app
        self.title("Heart Disease Prediction")
        self.geometry("550x700")
        self.resizable(False, False)

        # Themed style
        self.style= ttk.Style(self)
        self.style.theme_use('clam')  # Can be changed to the following theme---> 'clam', 'alt', 'default', 'classic'

        # Main Frame
        main_frame= ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill= "both")

        # Configure columns in main_frame to expand
        main_frame.grid_columnconfigure(0, weight=0)    # Label column, no expansion
        main_frame.grid_columnconfigure(1, weight=1)    # Widget column, allows expansion
        main_frame.grid_columnconfigure(2, weight=0)    # Value label column, no expansion

        # App Title
        app_title = ttk.Label(main_frame, text="Heart Disease Prediction App", font=("Segoe UI", 16, "bold"))
        app_title.grid(row=0, column=0, columnspan=3, pady=10)

        app_description = ttk.Label(main_frame, text="This app uses a machine learning model to predict the likelihood of heart disease based on patient data.", wraplength=400)
        app_description.grid(row=1, column=0, columnspan=3, pady=5)

        # Input Widget
        self.inputs= {}                   # Storing all inputs in widget for easy access. 
        self.create_widget(main_frame)

        # Prediction Button
        predict_button= ttk.Button(main_frame, text="Predict Heart Disease Risk", command= self.predict)
        predict_button.grid(row=17, column=0, columnspan=3, pady=(5,15))

        # Frame to hold result and probability together
        result_frame = tk.Frame(main_frame)
        result_frame.grid(row=16, column=0, columnspan=3, pady=(10, 5))
        # Displays the result
        self.result_label= ttk.Label(result_frame, text="", font=("Segoe UI", 12, "bold"))
        self.result_label.pack(anchor='center')

        self.proba_label= ttk.Label(result_frame, text="", font=("Segoe UI",10))
        self.proba_label.pack(anchor='center')

        # Disclaimer
        disclaimer_label = ttk.Label(main_frame, text="Disclaimer: This is a demo application. Do not use for actual medical diagnostics. Consult a healthcare professional for any medical concerns.", wraplength=400, font=("Segoe UI", 8, "italic"))
        disclaimer_label.grid(row=18, column=0, columnspan=3, pady=(10,5))

    def create_widget(self, parent_frame):
        """
            Creates and place all the input widget in a parent frame.
        """
        # Define input value and properties
        # (Label, key_name, widget_type, options/range, default_value)--->tkinter format for inputs
        widget_definitions= [
            ("Age", "age", "slider", (20,100), 54),
            ("Sex", "sex", "combo", ["Male", "Female"], "Male"),
            ("Chest Pain Type (cp)", "chest", "combo", [0, 1, 2, 3], 0),
            ("Resting Blood Pressure (trestbps)", "trestbps", "slider", (90, 200), 131),
            ("Serum Cholestrol (chol)", "chol", "slider", (126, 564), 246),
            ("Fasting Blood Sugar > 120 (fbs)", "fbs", "combo", ["True", "False"], "False"),
            ("Resting ECG (restecg)", "restecg", "combo", [0, 1, 2], 1),
            ("Max Heart Rate (thalach)", "thalach", "slider", (71, 202),149),
            ("Exercise Induced Angina (exang)", "exang", "combo", ["Yes", "No"], "No"),
            ("ST Depression (oldpeak)", "oldpeak", "slider", (0.0, 6.2), 1.0),
            ("Slope of ST Segment (slope)", "slope", "combo", [0, 1, 2], 1),
            ("Major Vessels Colored (ca)", "ca","combo", [0, 1, 2, 3], 0),
            ("Thalassemia (thal)", "thal", "combo", [0, 1, 2, 3], 2)
        ]
        
        # Loop through definitions to create widgets
        for i, (label_text, key, widget_type, options, default) in enumerate(widget_definitions):
            label = ttk.Label(parent_frame, text=label_text)
            label.grid(row=i+2, column=0, sticky="w", pady=2)

            if widget_type == "slider":
                if key == 'oldpeak':
                    var = tk.DoubleVar(value=default)
                    display_var = tk.StringVar(value=f"{default:.2f}")
                    value_label = ttk.Label(parent_frame, textvariable=display_var, width=6)
                    widget = ttk.Scale(parent_frame, from_=options[0], to=options[1], variable=var, orient="horizontal", command=lambda val, dv=display_var: dv.set(f"{float(val):.2f}"))
                else:
                    var = tk.IntVar(value=int(default))
                    value_label = ttk.Label(parent_frame, textvariable=var, width=6)
                    widget = ttk.Scale(parent_frame, from_=options[0], to=options[1], value=default, orient="horizontal", command=lambda val, v=var: v.set(round(float(val))))
                widget.grid(row=i+2, column=1, sticky="ew", padx=5, pady=3)
                value_label.grid(row=i+2, column=2, padx=5,pady=3)

            elif widget_type == "combo":
                var = tk.StringVar(value=default)
                widget = ttk.Combobox(parent_frame, textvariable=var, values=options, state="readonly")
                widget.grid(row=i+2, column=1, sticky="ew", padx=5, pady= 3)

            self.inputs[key] = var

            # Configures column weights for proper resizing
            parent_frame.columnconfigure(1,weight=1)

    def predict(self):
        """
        Gathers input, preprocesses it, makes a prediction, and updates the UI.
        """
        if not model_loaded:
            messagebox.showwarning("Warning","Model artifact not loaded.")
            return
        
        try:
            input_data= {
                'age': self.inputs['age'].get(),
                'sex': 1 if self.inputs['sex'].get() == 'Male' else 0,
                'cp': int(self.inputs['chest'].get()),
                'trestbps': self.inputs['trestbps'].get(),
                'chol': self.inputs['chol'].get(),
                'fbs': 1 if self.inputs['fbs'].get() == "True" else 0,
                'restecg': int(self.inputs['restecg'].get()),
                'thalach': self.inputs['thalach'].get(),
                'exang': 1 if self.inputs['exang'].get() == "Yes" else 0,
                'oldpeak': self.inputs['oldpeak'].get(),
                'slope': int(self.inputs['slope'].get()),
                'ca': int(self.inputs['ca'].get()),
                'thal': int(self.inputs['thal'].get())
            }

            input_df= pd.DataFrame([input_data])

            prediction = pipeline.predict(input_df)
            prediction_proba = pipeline.predict_proba(input_df)

            if prediction[0] ==1:
                self.result_label.config(text= "Low Risk of Heart Disease", foreground="green")
                self.proba_label.config(text= f"Prediction Probability: {prediction_proba[0][1]*100:.2f}%")
            else:
                self.result_label.config(text= "High Risk of Heart Disease", foreground="red")
                self.proba_label.config(text= f"Prediction Probability: {prediction_proba[0][0]*100:.2f}%")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")

# Run the App
if __name__ == "__main__":
    if model_loaded:
        app= HeartDiseaseApp()
        app.mainloop()
    else:
        print("Application not started due to model loading error.")