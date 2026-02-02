from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from churn.features.preprocessing import build_preprocessor


def logistic_regression_model():
    """
    Returns a GridSearchCV object
    for Logistic Regression.
    """
    pipe = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    param_grid = [
        # L2 and no penalty - works with lbfgs
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": [None, "l2"],
            "model__solver": ["lbfgs"],
            "model__class_weight": [None, "balanced"],
        },
        # L1 penalty - needs saga
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l1"],
            "model__solver": ["saga"],
            "model__class_weight": [None, "balanced"],
        },
        # Elasticnet - needs saga + l1_ratio
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["elasticnet"],
            "model__solver": ["saga"],
            "model__l1_ratio": [0.25, 0.5, 0.75],
            "model__class_weight": [None, "balanced"],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    return search
