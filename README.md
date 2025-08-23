# Heart Disease Prediction Project ❤️

This project develops a machine learning model to predict the likelihood of heart disease based on various patient health indicators. The goal is to create a robust and interpretable model that can assist in early risk assessment.

---

## Table of Contents
1.  Introduction
2.  Methodology
    * Data Exploration & Visualization
    * Data Preprocessing
    * Model Training & Hyperparameter Tuning
    * Model Evaluation
3.  Project Structure
4.  Getting Started
    * Prerequisites
    * Installation
    * Running the Scripts
    * Running the Tkinter App
    * Running the Streamlit App
5.  Results
6.  Usage
7.  Conclusion

---

## 1. Introduction

Heart disease remains a leading cause of mortality worldwide. Early prediction and intervention can significantly improve patient outcomes. This project leverages a dataset containing various health parameters to build a machine learning model capable of predicting heart disease risk.

**Objective**: The primary objective of this project was to conduct a comprehensive analysis to **find the best-tuned classic machine learning model** for heart disease prediction. This involved training and meticulously evaluating various algorithms. Through this rigorous process, the **K-Nearest Neighbors (KNN) model** consistently demonstrated superior performance, ultimately standing out as the most robust and accurate solution for this problem.

---

## 2. Methodology

The project follows a standard machine learning pipeline:

### Data Exploration & Visualization
* The raw dataset (`heart.csv`) was loaded and thoroughly explored.
* Initial analysis confirmed **no missing values** across 14 attributes.
* **Histograms** and **box plots** were used to understand the distribution of numerical features and identify potential outliers.
* **Countplots** were generated for categorical features to visualize their distribution against the target variable.
* A **correlation heatmap** was created to identify relationships between features.
* A **pairplot** was used to visualize pairwise relationships among key numerical features, colored by the target variable.

### Data Preprocessing
* A `ColumnTransformer` was used to apply different preprocessing steps to numerical and categorical features within a `Pipeline`.
* **Numerical features** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) were scaled using `MinMaxScaler`.
* **Categorical features** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) were encoded using `OneHotEncoder` with `drop='first'` to avoid multicollinearity.

### Model Training & Hyperparameter Tuning
* The dataset was split into **training (60%)**, **validation (20%)**, and **final unseen test (20%)** sets to ensure robust evaluation and prevent data leakage.
* Several **classic machine learning models** were evaluated for **baseline performance** on the validation set:
    * Logistic Regression
    * K-Nearest Neighbors (KNN)
    * Support Vector Machine (SVM)
    * Naive Bayes
    * Decision Tree
    * Random Forest
    * Gradient Boosting
    * XGBoost
* Each of these models underwent **hyperparameter tuning** using `GridSearchCV` on the training data.
* The **best-tuned version** of each model (including its preprocessor) was saved as a `.pkl` file in the `models/tuned/` directory.

### Model Evaluation
* The **K-Nearest Neighbors (KNN)** model was selected as the best-performing model based on its superior accuracy and AUC score during the tuning phase.
* A final, rigorous evaluation of the **KNN pipeline** was performed on the **completely unseen test data**.
* The model achieved an impressive **accuracy of 0.9902** on this unseen data, with excellent precision, recall, and F1-scores.

---

## 3. Project Structure

heart-disease-prediction/
├── app/
│   ├── app.py                   # Streamlit web application
│   ├── app_1.py                 # Another Streamlit web application (if applicable)
│   └── tkinter.py               # Tkinter GUI application
├── data/
│   ├── processed/
│   │   └── final_test_data.csv  # Unseen test data for final evaluation
│   └── raw/
│       └── heart.csv            # Original raw dataset
├── models/
│   ├── tuned/                   # Contains all hyper-tuned model pipelines
│   │   ├── decision_tree.pkl
│   │   ├── gradient_boosting.pkl
│   │   ├── knn.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── naive_bayes.pkl
│   │   ├── random_forest.pkl
│   │   ├── svm.pkl
│   │   └── xgboost.pkl
│   └── final_pipeline_knn.pkl   # The chosen best model for the Tkinter/Streamlit apps
├── notebooks/
│   ├── data_exploration.ipynb   # Detailed EDA and visualizations
│   ├── testing.ipynb            # Final comprehensive evaluation of all models
│   └── training.ipynb           # Model training, tuning, and selection
├── reports/
│   ├── figures/                 # Saved plots from visualization script
│   └── metrics/                 # Tuning results (e.g., tuning_results_knn.json)
└── scripts/
├── model_evaluation.py      # Script to evaluate the final KNN model
├── model_training.py        # Script to train and save the final KNN model
├── model_utils.py           # Helper functions (load/save data/models)
├── preprocessing.py         # Data preprocessing pipeline definition
└── visualisation.py         # Script to generate and save key visualizations


---

## 4. Getting Started

### Prerequisites
* Python 3.8+
* `pip` (Python package installer)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/heart-disease-prediction.git](https://github.com/your-username/heart-disease-prediction.git)
    cd heart-disease-prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Install all required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt` content:**
    ```
    # Core libraries for the ML model and app functionality
    numpy==1.26.4
    pandas==1.5.3
    scikit-learn==1.5.1
    joblib==1.2.0
    xgboost==1.7.5

    #---------------------------------------------------------------

    #Library for the web application
    streamlit==1.35.0

    #---------------------------------------------------------------

    #Libraries for development, analysis, and visualisation
    matplotlib==3.7.1
    seaborn==0.12.2
    jupyterlab==4.2.3
    ```

### Running the Scripts

To train and evaluate the final KNN model:

1.  **Train the final model:** This script will train the KNN model, perform hyperparameter tuning, and save the best pipeline as `final_pipeline_knn.pkl`. It also saves the `final_test_data.csv` for evaluation.
    ```bash
    python scripts/model_training.py
    ```

2.  **Evaluate the final model:** This script will load the `final_pipeline_knn.pkl` and evaluate its performance on the unseen test data.
    ```bash
    python scripts/model_evaluation.py
    ```

### Running the Tkinter App
To launch the interactive prediction application:

1.  Ensure you have run `scripts/model_training.py` at least once to create the `final_pipeline_knn.pkl` model.
2.  Run the Tkinter app:
    ```bash
    python app/tkinter.py
    ```
    This will open a GUI window where you can input patient data and get predictions.

### Running the Streamlit App
To launch the interactive Streamlit web application:

1.  Ensure you have run `scripts/model_training.py` at least once to create the `final_pipeline_knn.pkl` model.
2.  Run the Streamlit app (assuming `app.py` is your main Streamlit file):
    ```bash
    streamlit run app/app.py
    ```
    This will open the Streamlit app in your web browser. If you have `app_1.py` as another Streamlit app, you can run `streamlit run app/app_1.py` accordingly.

---

## 5. Results

The **K-Nearest Neighbors (KNN)** model was chosen as the best model. On a completely unseen test dataset, it achieved:

* **Accuracy:** **0.9902**
* **AUC Score:** **0.9994**
* **Precision (Class 0 / No Heart Disease):** 0.98
* **Recall (Class 0 / No Heart Disease):** 1.00
* **F1-Score (Class 0 / No Heart Disease):** 0.99
* **Precision (Class 1 / Heart Disease):** 1.00
* **Recall (Class 1 / Heart Disease):** 0.98
* **F1-Score (Class 1 / Heart Disease):** 0.99

This indicates a highly robust model capable of accurately classifying heart disease risk.

---

## 6. Usage

The Tkinter and Streamlit applications allow users to:
1.  Adjust various patient health parameters using sliders and dropdowns.
2.  Click the "Predict Heart Disease Risk" button to get a prediction.
3.  View the predicted risk (Low Risk / High Risk) and the associated probability.

**Disclaimer**: This application is for demonstration purposes only and should **not** be used for actual medical diagnostics. Always consult a healthcare professional for any medical concerns.

---

## 7. Conclusion

This project successfully developed and evaluated a machine learning pipeline for heart
