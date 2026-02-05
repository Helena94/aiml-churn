from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from churn.features.preprocessing import build_preprocessor


def build_logistic_regression_model(param_grid=None, SearchCVConfig=None, StratifiedKFoldConfig=None):
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
    param_grid = param_grid or [
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

    cv = StratifiedKFold(
        n_splits=StratifiedKFoldConfig.get("n_splits", 5) if StratifiedKFoldConfig else 5,
        shuffle=StratifiedKFoldConfig.get("shuffle", True) if StratifiedKFoldConfig else True,
        random_state=StratifiedKFoldConfig.get("random_state", 42) if StratifiedKFoldConfig else 42,
    )

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=SearchCVConfig.get("scoring", "roc_auc") if SearchCVConfig else "roc_auc",
        cv=SearchCVConfig.get("cv", cv) if SearchCVConfig else cv,
        n_jobs=SearchCVConfig.get("n_jobs", -1) if SearchCVConfig else -1,
        verbose=SearchCVConfig.get("verbose", 1) if SearchCVConfig else 1,
    )

    return search
