"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Online Shoppers Purchasing Intention.
The agent can modify:
1. Feature selection (which columns to include)
2. Hyperparameters (max_iter, learning_rate, etc.)
3. Preprocessing steps (Scaling vs. No Scaling)
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_model():
    """Return an sklearn Pipeline. This is what the agent improves."""
    
    # --- 1. FEATURE SELECTION ---
    # The agent can modify these lists to include/exclude specific columns
    categorical_cols = [
        'VisitorType', 'Weekend', 'OperatingSystems', 
        'Browser', 'Region', 'TrafficType'
    ]
    
    numerical_cols = [
        'Administrative', 'Administrative_Duration', 
        'Informational', 'Informational_Duration', 
        'ProductRelated', 'ProductRelated_Duration', 
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
    ]

    # --- 2. FEATURE ENGINEERING / PREPROCESSING ---
    # The agent can swap StandardScaler for RobustScaler if outliers are an issue
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # --- 3. HYPERPARAMETER TUNING ---
    # The agent can change these values to optimize ROC AUC and Recall
    model_params = {
        'max_iter': 250,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_samples_leaf': 25,
        'l2_regularization': 0.1,
        'class_weight': 'balanced',  # Crucial for the "Recall Floor"
        'random_state': 42
    }

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingClassifier(**model_params)),
    ])
