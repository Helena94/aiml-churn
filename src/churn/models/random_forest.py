from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from churn.features.preprocessing import build_preprocessor


def random_forest_model():
    """
    Returns a RandomizedSearchCV object
    for Random Forest.
    """
    pipe = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    param_distributions = {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    return search   