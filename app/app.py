# This is a app for the heart-disease-prediction

import streamlit as st
import pandas as pd
import joblib 
import os

# Setting up the path
app_dir= os.path.dirname(os.path.abspath(__file__))
project_dir= os.path.dirname(app_dir)
model_path= os.path.join(project_dir, "models", "final_model.pkl")
preprocessor_path= os.path.join(project_dir, "models", "preprocessor.pkl")

# Loading the artifacts
@st.cache_resource   ## To load the model and preprocessor only once
def load_artifacts():
    """Loads the pre-trained model and preprocessor."""
    try:
        model= joblib.load(model_path)
        preprocessor= joblib.load(preprocessor_path)
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please make sure the 'final_model.pkl' and 'preprocessor.pkl' file are in the 'models' directory.")
        return None, None
    
model, preprocessor= load_artifacts()

# --- APP LAYOUT ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
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
    fbs= st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ('False', 'True'))
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
    exang_val = 1 if exang == 'True' else 0

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
if model and preprocessor:
    # Displays the user's input data
    st.subheader("Patient's Input Data")  
    st.write(input_df)

    # Prediction Button
    if st.sidebar.button('Predict Heart Disease Risk'):
        try:
            # Preprocess the user input
            preprocessed_input= preprocessor.transform(input_df)

            # Make a prediction
            prediction= model.predict(preprocessed_input)
            prediction_proba= model.predict_proba(preprocessed_input)

            # Display the result
            st.header("Prediction Result")
            if prediction[0] == 1:
                st.error("High Risk: The model predicts a high likelihood of heart disease.")
                st.write(f"**Prediction Probability:** {prediction_proba[0][1]*100:.2f}")
            else:
                st.success("Low Risk: The model predicts a low likelihood of heart disease.")
                st.write(f"**Prediction Probability** {prediction_proba[0][0]*100:.2f}")

        except Exception as e:
            st.error(f"An error occured during prediction: {e}")
else:
    st.warning("Model artifacts not loaded. Prediction is unavailable.")

st.sidebar.markdown("----")
st.sidebar.info("This is a demo application. Do not use for actual medical diagnostics.")