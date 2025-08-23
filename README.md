# Heart Disease Prediction Project

This project applies data analysis and machine learning to predict the likelihood of heart disease.  
It uses the **Heart Disease dataset (`heart.csv`)** derived from the Kaggle dataset "[Heart Disease Dataset]
---

##  Dataset Overview

The dataset consists of **1,025 patient records** with **14 medical attributes** used to predict heart disease.  

**Key Statistics:**  
- **Age:** 29 – 77 years (mean ≈ 54.4)  
- **Cholesterol:** 126 – 564 mg/dl (mean ≈ 246)  
- **Resting Blood Pressure:** 94 – 200 mm Hg (mean ≈ 132)  
- **Maximum Heart Rate (thalach):** 71 – 202 bpm (mean ≈ 149)  
- **Oldpeak (ST depression):** 0 – 6.2 (mean ≈ 1.07)  

**Features include:**  
- `age` → Age of patient  
- `sex` → Gender (1 = male, 0 = female)  
- `cp` → Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal, 3 = asymptomatic)  
- `trestbps` → Resting blood pressure (mm Hg)  
- `chol` → Serum cholesterol (mg/dl)  
- `fbs` → Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
- `restecg` → Resting electrocardiographic results (0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy)  
- `thalach` → Maximum heart rate achieved  
- `exang` → Exercise induced angina (1 = yes, 0 = no)  
- `oldpeak` → ST depression induced by exercise  
- `slope` → Slope of the peak exercise ST segment (0 = up, 1 = flat, 2 = down)
- `ca` → Number of major vessels (0–3) colored by fluoroscopy  
- `thal` → Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)  
- `target` → Presence of heart disease (1 = disease, 0 = no disease)  

---

##  Project Workflow

1. **Exploratory Data Analysis** (`data_exploration.ipynb`)  
   - Distribution plots of key features  
   - Correlation heatmap  
   - Visual comparison of target vs. predictors  

2. **Model Training** (`training.ipynb`)  
   - Trained multiple models (Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
   - Hyper parameter tuned all the model
   - Selected **K-Nearest Neighbors** as the best-performing model  

3. **Model Testing** (`testing.ipynb`)  
   - Evaluated the hyperparameter tuned KNN  model on unseen test data  

---

##  Results (Final Model – Tuned KNN)

- **Accuracy**: 99.02%  
- **Precision**:  
  - Class 0 (No Disease): 0.98  
  - Class 1 (Disease): 1.00 
- **Recall**:  
  - Class 0: 1.00  
  - Class 1: 0.98  
- **F1-Score**:  
  - Class 0: 0.99  
  - Class 1: 0.99  

**Confusion Matrix:**  
```
                 Predicted No   Predicted Yes
Actual No              100             0
Actual Yes              2             103
```

---

##  Conclusion & Future Work

This project demonstrates how machine learning can be effectively used to predict the likelihood of heart disease based on patient medical data.  

 The tuned **K-Nearest Neighbors model** achieved **99.02% accuracy**, with strong precision and recall scores.  
 The model highlights important health attributes like chest pain type, cholesterol levels, and maximum heart rate as strong predictors.  

**Future Improvements:**  
- Experiment with deep learning approaches (e.g., Neural Networks).  
- Perform feature engineering to capture more complex interactions.   
- Deploy the model as a **web application** or **API** for real-world usability.  
