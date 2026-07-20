from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from churn.training.evaluation import evaluate_model
from churn.training.train import train_model


def run_with_mlflow(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
        config: dict,
):
    mlflow.set_tracking_uri("sqlite:///" + str(Path(__file__).resolve().parents[3] / "mlflow.db"))
    experiment = mlflow.set_experiment("churn-training")

    with mlflow.start_run(run_name=model_name, experiment_id=experiment.experiment_id):
        model = train_model(X_train, y_train, model_name=model_name, config=config)

        # Log the full config for provenance/reproducibility (dict -> artifact)
        mlflow.log_dict(config, "config.yml")

        # Log the searched hyperparameter space (GridSearchCV `param_grid` or
        # RandomizedSearchCV `param_distributions`). `search.` prefix avoids
        # colliding with best_params_ keys like `model__max_depth`.
        # (kmeans uses the misspelled `params_grid` key — matched here, not renamed)
        search_space = (config.get("param_distributions") or config.get("param_grid")
                        or config.get("params_grid") or {})
        # param_grid may be a list of dicts (disjoint grids, e.g. logistic_regression);
        # index them so conflicting keys across grids don't collide.
        grids = search_space if isinstance(search_space, list) else [search_space]
        for i, grid in enumerate(grids):
            prefix = f"search.{i}." if len(grids) > 1 else "search."
            mlflow.log_params({f"{prefix}{k}": v for k, v in grid.items()})

        # Log CV / search settings (scoring, n_iter, n_splits, random_state, ...)
        mlflow.log_params({f"cv.{k}": v for k, v in config.get("SearchCVConfig", {}).items()})
        mlflow.log_params({f"cv.{k}": v for k, v in config.get("StratifiedKFoldConfig", {}).items()})

        # Log best hyperparameters found by CV search
        if hasattr(model, "best_params_"):
            mlflow.log_params(model.best_params_)

        mlflow.log_param("model_name", model_name)

        # Evaluate
        model_type = "clustering" if model_name == "kmeans" else "classification"
        metrics = evaluate_model(model, X_test, y_test, model_type)

        # MLflow only accepts scalar metrics — skip confusion_matrix list
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(scalar_metrics)

        # Log the trained model as an artifact
        mlflow.sklearn.log_model(model, name="model-{}".format(model_name))

    return model, metrics
