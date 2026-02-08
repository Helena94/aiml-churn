from churn.models.kmeans import build_kmeans_model
from churn.models.logistic_regression import build_logistic_regression_model
from churn.models.random_forest import build_random_forest_model
from churn.models.xgboost import build_xgboost_model


def train_model(X_train, y_train, model_name, config):
    """
    Trains the given model using the training data.
    Returns the trained model.
    """
    if model_name == "logistic_regression":
        model = build_logistic_regression_model(param_grid=config.get("param_grid"),
                                                SearchCVConfig=config.get("SearchCVConfig"),
                                                StratifiedKFoldConfig=config.get("StratifiedKFoldConfig"))
        model.fit(X_train, y_train)
    elif model_name == "random_forest":
        model = build_random_forest_model(param_distributions=config.get("param_distributions"),
                                          SearchCVConfig=config.get("SearchCVConfig"),
                                          StratifiedKFoldConfig=config.get("StratifiedKFoldConfig"))
        model.fit(X_train, y_train)
    elif model_name == "xgboost":
        model = build_xgboost_model(param_distributions=config.get("param_distributions"),
                                    SearchCVConfig=config.get("SearchCVConfig"),
                                    StratifiedKFoldConfig=config.get("StratifiedKFoldConfig"))
        model.fit(X_train, y_train)
    elif model_name == "kmeans":
        model = build_kmeans_model(params_grid=config.get("params_grid"), SearchCVConfig=config.get("SearchCVConfig"))
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model
