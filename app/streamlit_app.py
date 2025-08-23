# This is a app for the heart-disease-prediction

import streamlit as st
import pandas as pd
import joblib 
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Setting up the path
try:
    app_dir= os.path.dirname(os.path.abspath(__file__))
    project_dir= os.path.dirname(app_dir)
    model_path= os.path.join(project_dir, "models", "final_pipeline_knn.pkl")

    # Loading the artifacts
    @st.cache_resource   ## To load the model and preprocessor only once
    def load_artifacts():
        """Loads the pre-trained model and preprocessor."""
        try:
            pipeline= joblib.load(model_path)
            return pipeline 
        except FileNotFoundError as e:
            st.error(f"Error loading model artifacts: {e}")
            st.info("Please make sure the 'final_pipeline_knn.pkl' file is in the 'models' directory.")
            return None
        
    pipeline= load_artifacts()
except Exception as e:
    st.error(f"An error occured during the startup: {e}")
    pipeline= None

# --- APP LAYOUT ---

st.title("Heart Disease Prediction App")
st.markdown("This app uses a machine learning model to predict the likelihood of heart disease bsaed on the patient data.")

# Users Input Layout
st.sidebar.header("Patient Input Features")

def get_user_input():
    age= st.sidebar.slider('Age', 20, 100, 54)
    sex= st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp= st.sidebar.selectbox('Chest Pain Type (cp)', (0, 1, 2, 3))
    trestbps= st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 131)
    chol= st.sidebar.slider('Serum Cholestrol in mg/dl (chol)', 126, 564, 246 )
    fbs= st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ('True', 'False'))
    restecg= st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', (0, 1, 2))
    thalach= st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 149)
    exang= st.sidebar.selectbox('Exercise Induced Angina (exang)', ('Yes', 'No'))
    oldpeak= st.sidebar.slider('ST depression induced by exercise relative to rest (oldpeak)', 0.0, 6.2, 1.0)
    slope= st.sidebar.selectbox('Slope of the peak exercise ST segment (slope)', (0, 1, 2))
    ca= st.sidebar.selectbox('Number of major values colored by flourosopy (ca)', (0, 1, 2, 3))
    thal= st.sidebar.selectbox('Thalassemia', (0, 1, 2, 3))

    # Conversion of categorical inputs to numerical format which the model expects
    sex_val = 1 if sex == 'Male' else 0
    fbs_val = 1 if fbs == 'True' else 0
    exang_val = 1 if exang == 'Yes' else 0

    # Creating a dictionary for the inputs
    input_data={
        'age': age,
        'sex': sex_val,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_val,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang_val,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # Converting the dictionary to a DataFrame
    input_df= pd.DataFrame([input_data])
    return input_df

input_df= get_user_input()

# --- PREDICTION & DISPLAY ---
if pipeline:
    # Displays the user's input data
    st.subheader("Patient's Input Data")  
    st.write(input_df)

    # Prediction Button
    if st.sidebar.button('Predict Heart Disease Risk'):
        try:
            # Pipeline handles both preprocessing and prediction
            prediction= pipeline.predict(input_df)
            prediction_proba= pipeline.predict_proba(input_df)

            # Display the result
            st.header("Prediction Result")
            if prediction[0] == 1:
                st.success("Low Risk: The model predicts a low likelihood of heart disease.")
                st.write(f"**Prediction Probability (No Heart Disease):** {prediction_proba[0][1]*100:.2f}%")
                
                # --- Post-Prediction Information for Low Risk ---
                st.markdown("---")
                st.subheader("General Recommendations")
                st.info("Great! Continue to maintain a healthy lifestyle. Regular exercise, a balanced diet, and routine check-ups are always beneficial for heart health.")

            else:
                st.error("High Risk: The model predicts a high likelihood of heart disease.")
                st.write(f"**Prediction Probability (Heart Disease):** {prediction_proba[0][0]*100:.2f}%")
                
                # --- Post-Prediction Information for High Risk ---
                st.markdown("---")
                st.subheader("General Recommendations")
                st.warning("It's recommended to consult a healthcare professional for a comprehensive evaluation. They can provide personalized advice and further tests if needed. Maintaining a healthy diet, regular physical activity, and managing stress are crucial for heart health.")

            # --- About the Model ---
            st.markdown("---")
            st.subheader("About This Prediction")
            st.info("This prediction was made using a **K-Nearest Neighbors (KNN)** machine learning model. The model achieved an accuracy of **99.02%** on unseen test data.")

        except Exception as e:
            st.error(f"An error occured during prediction: {e}")
else:
    st.warning("Model pipeline not loaded. Prediction is unavailable.")

st.sidebar.markdown("----")
st.sidebar.info("This is a demo application. Do not use for actual medical diagnostics.")