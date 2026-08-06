import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from churn.training.evaluation import evaluate_model
from churn.training.train import train_model


def mlflow_tracking_uri() -> str:
    """Remote tracking server if MLFLOW_TRACKING_URI is set, else the project-local SQLite store.

    A container's filesystem does not survive the run, so any deployment needs a remote
    server; the SQLite default keeps local runs working with no configuration.
    """
    return os.getenv("MLFLOW_TRACKING_URI") or (
        "sqlite:///" + str(Path(__file__).resolve().parents[3] / "mlflow.db") + "?timeout=30"
    )


def register_champion(
        run_id: str,
        artifact_name: str,
        registry_name: str,
        alias: str = "champion",
) -> int:
    """
    Register a logged model under `registry_name` and point `alias` at the new version.

    Args:
        run_id: MLflow run that logged the model.
        artifact_name: Model name used at log time (the `model-<name>` artifact path).
        registry_name: Registered model name to create/extend.
        alias: Alias to move onto the new version.

    Returns:
        The registered model version number.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    version = mlflow.register_model(f"runs:/{run_id}/model-{artifact_name}", registry_name)
    MlflowClient().set_registered_model_alias(registry_name, alias, version.version)
    return int(version.version)


def run_with_mlflow(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
        config: dict,
):
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    experiment = mlflow.set_experiment("churn-training")

    with mlflow.start_run(run_name=model_name, experiment_id=experiment.experiment_id) as run:
        run_id = run.info.run_id
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

        # Save the model locally, then upload it under the *run's* artifact path.
        #
        # Not `log_model(name=...)`: in MLflow 3 that creates a "logged model" entity stored
        # outside the run's artifact tree, and register_model("runs:/<id>/<name>") then has to
        # look it up through the logged-models API (see mlflow/tracking/_model_registry/
        # fluent.py). A tracking server that does not implement that API — DagsHub does not —
        # fails with "Unable to find a logged_model with artifact_path ...". Writing the model
        # under the run makes register_model take its classic branch, which every MLflow-
        # compatible server supports.
        #
        # serialization_format is explicit because newer MLflow defaults to "skops", which
        # refuses to serialize a SearchCV: it carries a scorer and a StratifiedKFold that
        # skops treats as untrusted types.
        artifact_path = "model-{}".format(model_name)
        with tempfile.TemporaryDirectory() as tmp:
            local_model_path = Path(tmp) / artifact_path
            mlflow.sklearn.save_model(
                model,
                str(local_model_path),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            mlflow.log_artifacts(str(local_model_path), artifact_path=artifact_path)

    return model, metrics, run_id
