# This is a tkinter app for heart-disease-prediction

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
import os

from tkinter import messagebox, ttk

# Setting up the path
try:
    app_dir= os.path.dirname(os.path.abspath(__file__))
    project_dir= os.path.dirname(app_dir)
    model_path= os.path.join(project_dir, "models", "final_model.pkl")
    preprocessor_path= os.path.join(project_dir, "models", "preprocessor.pkl")

    # Loading the  artifacts
    model= joblib.load(model_path)
    preprocessor= joblib.load(preprocessor_path)
except FileNotFoundError as e:
    messagebox.showerror("Error loading model artifacts: {e}")
    print("Please make sure the 'final_model.pkl' and 'preprocessor.pkl' file are in the 'models' directory.")
    model= None
    preprocessor= None
except Exception as e:
    messagebox.showerror("Error",f"An unexpected error occured during the model loading: {e}")
    model= None
    preprocessor= None

# -- APP LAYOUT --
class HeartDiseaseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Size of the app
        self.title("Heart Disease Prediction")
        self.geometry("400x650")
        self.resizable(False, False)

        # Themed style
        self.style= ttk.Style(self)
        self.style.theme_use('clam')  # Can be changed to the following theme---> 'clam', 'alt', 'default', 'classic'

        # Main Frame
        main_frame= ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill= "both")

        # Input Widget
        self.inputs= {}                   # Storing all inputs in widget for easy access. 
        self.create_widget(main_frame)

        # Prediction Button
        predict_button= ttk.Button(main_frame, text="Predict Heart Disease Risk", command= self.predict)
        predict_button.grid(row=13, column=0, columnspan=2, pady=20)

        # Displays the result
        self.result_label= ttk.Label(main_frame, text="", font=("Segoe UI", 12, "bold"))
        self.result_label.grid(row= 14, column=0, columnspan=2)

        self.proba_label= ttk.Label(main_frame, text="", font=("Segoe UI",10))
        self.proba_label.grid(row=15, column=0, columnspan=2)

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
            label= ttk.Label(parent_frame, text= label_text)
            label.grid(row=i, column=0, sticky="w", pady=5)

            if widget_type == "slider":
                if key == 'oldpeak':
                    var= tk.DoubleVar(value=default)
                    display_var= tk.StringVar(value=f"{default:.2f}")
                    value_label= ttk.Label(parent_frame, textvariable=display_var, width=6)
                    widget= ttk.Scale(parent_frame, from_=options[0], to=options[1], variable=var, orient="horizontal", command=lambda val, dv=display_var: dv.set(f"{float(val):.2f}"))
                else:
                    var= tk.IntVar(value=int(default))
                    value_label= ttk.Label(parent_frame, textvariable=var, width=6)
                    widget= ttk.Scale(parent_frame, from_=options[0], to=options[1], value=default, orient="horizontal", command=lambda val, v=var: v.set(round(float(val))))
                    
                widget.grid(row=i, column=1, sticky="ew", padx=5)
                value_label.grid(row=i, column=2, padx=5)
                self.inputs[key]= var

            elif widget_type == "combo":
                var= tk.StringVar(value=default)
                widget= ttk.Combobox(parent_frame, textvariable=var, values=options, state="readonly")
                widget.grid(row=i, column= 1, sticky= "ew", padx=5)
                self.inputs[key]= var

            # Configures column weights for proper resizing
            parent_frame.columnconfigure(1,weight=1)

    def predict(self):
        """
        Gathers input, preprocesses it, makes a prediction, and updates the UI.
        """
        if not model or not preprocessor:
            messagebox.showwarning("Warning","Model artifact not loaded.")
            return
        
        try:
            input_data= {
                'age': self.inputs['age'].get(),
                'sex': 1 if self.inputs['sex'].get() == 'Female' else 0,
                'cp': int(self.inputs['chest'].get()),
                'trestbps': self.inputs['trestbps'].get(),
                'chol': self.inputs['chol'].get(),
                'fbs': 1 if self.inputs['fbs'] == "True" else 0,
                'restecg': int(self.inputs['restecg'].get()),
                'thalach': self.inputs['thalach'].get(),
                'exang': 1 if self.inputs['exang'] == "Yes" else 0,
                'oldpeak': self.inputs['oldpeak'].get(),
                'slope': int(self.inputs['slope'].get()),
                'ca': int(self.inputs['ca'].get()),
                'thal': int(self.inputs['thal'].get())
            }

            input_df= pd.DataFrame([input_data])

            preprocessed_input= preprocessor.transform(input_df)
            prediction= model.predict(preprocessed_input)
            prediction_proba= model.predict_proba(preprocessed_input)

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
    if model and preprocessor:
        app= HeartDiseaseApp()
        app.mainloop()



    