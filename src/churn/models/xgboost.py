from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from churn.features.preprocessing import build_preprocessor

def build_xgboost_model(param_distributions=None, SearchCVConfig=None, StratifiedKFoldConfig=None):
    """
    Returns a RandomizedSearchCV object for HistGradientBoostingClassifier.
    Drop-in replacement for XGBoost — no OpenMP dependency required.
    """
    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", HistGradientBoostingClassifier(random_state=42)),
    ])

    param_distributions = param_distributions or {
        "model__max_iter": [200, 500, 800],
        "model__max_depth": [3, 5, 7, None],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__min_samples_leaf": [10, 20, 30],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    }

    cv = StratifiedKFold(
        n_splits=StratifiedKFoldConfig.get("n_splits", 5) if StratifiedKFoldConfig else 5,
        shuffle=StratifiedKFoldConfig.get("shuffle", True) if StratifiedKFoldConfig else True,
        random_state=StratifiedKFoldConfig.get("random_state", 42) if StratifiedKFoldConfig else 42,
    )

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=SearchCVConfig.get("n_iter", 25) if SearchCVConfig else 25,
        scoring=SearchCVConfig.get("scoring", "roc_auc") if SearchCVConfig else "roc_auc",
        cv=SearchCVConfig.get("cv", cv) if SearchCVConfig else cv,
        n_jobs=SearchCVConfig.get("n_jobs", -1) if SearchCVConfig else -1,
        random_state=42,
        verbose=SearchCVConfig.get("verbose", 1) if SearchCVConfig else 1,
    )

    return search
