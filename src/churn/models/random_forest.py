from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

def build_random_forest_model(param_distributions=None, SearchCVConfig=None, StratifiedKFoldConfig=None):
    """
    Returns a RandomizedSearchCV object
    for Random Forest.
    """
    pipe = Pipeline(
        steps=[
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    param_distributions = param_distributions or {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": [None, "balanced"],
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
        random_state= SearchCVConfig.get("random_state", 42) if SearchCVConfig else 42,
        verbose= SearchCVConfig.get("verbose", 1) if SearchCVConfig else 1,
    )

    return search   