# This script is dedicated to preprocessing the raw heart disease data.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def create_preprocessor():
    """
    Creates and returns a scikit-learn ColumnTransformer for preprocessing.
    """

    categorical_features= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_features= ['age', 'trestbps', 'chol', 'thalach', 'olpeak']

    preprocessor= ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features)
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor
