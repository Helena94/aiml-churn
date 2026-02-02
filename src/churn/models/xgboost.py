



from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import Pipeline
from churn.features.preprocessing import build_preprocessor 

def xgboost_model():
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

    param_distributions = {
        "model__n_estimators": [200, 500, 800],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        random_state=42
    )

    return search
