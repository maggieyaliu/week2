"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Online Shoppers Purchasing Intention.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def build_model():
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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])
