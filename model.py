"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Online Shoppers Purchasing Intention.
The function build_model() must return an sklearn-compatible estimator.
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_model():
    """Return an sklearn Pipeline. This is what the agent improves."""
    
    # 1. Define feature groups (from the Online Shoppers dataset)
    categorical_cols = ['VisitorType', 'Weekend', 'OperatingSystems', 
                        'Browser', 'Region', 'TrafficType']
    
    numerical_cols = ['Administrative', 'Administrative_Duration', 'Informational', 
                      'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
                      'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

    # 2. Preprocessing: Scaling numbers and encoding categories
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # 3. Model: HistGradientBoosting is excellent for this dataset because it 
    # handles the non-linear relationship between PageValues and Revenue well.
    # The agent can modify these hyperparameters or swap the model entirely.
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingClassifier(
            max_iter=200, 
            max_depth=5, 
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced'  # Added to address the 0.34 recall floor
        )),
    ])
