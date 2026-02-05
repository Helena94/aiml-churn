



from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
from churn.features.preprocessing import build_preprocessor 

def build_xgboost_model(param_distributions=None, SearchCVConfig=None, StratifiedKFoldConfig=None):
    """
    Returns a RandomizedSearchCV object
    for XGBoost.
    """
    pipe = Pipeline(steps=[
        ("preprocessing", build_preprocessor()),
        ("model", XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        ))
    ])

    param_distributions = param_distributions or {
        "model__n_estimators": [200, 500, 800],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    }

    cv = StratifiedKFold(
        n_splits=StratifiedKFoldConfig.get("n_splits", 5) if StratifiedKFoldConfig else 5,
        shuffle=StratifiedKFoldConfig.get("shuffle", True) if StratifiedKFoldConfig else True,
        random_state=StratifiedKFoldConfig.get("random_state", 42) if StratifiedKFoldConfig else 42,
    )

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter= SearchCVConfig.get("n_iter", 25) if SearchCVConfig else 25,
        scoring=SearchCVConfig.get("scoring", "roc_auc") if SearchCVConfig else "roc_auc",
        cv= SearchCVConfig.get("cv", cv) if SearchCVConfig else cv,
        n_jobs= SearchCVConfig.get("n_jobs", -1) if SearchCVConfig else -1,
        random_state=42
    )

    return search
