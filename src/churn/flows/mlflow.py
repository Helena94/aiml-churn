import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from churn.training.evaluation import evaluate_model
from churn.training.train import train_model

# The metric champions are promoted on: every SearchCV's `best_score_`, i.e. the
# cross-validated score over the *training* folds. Deliberately not one of the held-out
# test metrics — those already pick the model *type* within a run, and ranking across runs
# on them too would let the same 20% holdout steer selection over and over until the
# reported score is optimistic. Logged under one key for all four models even though the
# underlying scorer differs (roc_auc for the classifiers, silhouette for kmeans): each
# registry only ever compares a model against its own history, so the scales never meet.
# Change `scoring` in a config and the versions logged before it stop being comparable.
SELECTION_METRIC = "cv_score"


def mlflow_tracking_uri() -> str:
    """Remote tracking server if MLFLOW_TRACKING_URI is set, else the project-local SQLite store.

    A container's filesystem does not survive the run, so any deployment needs a remote
    server; the SQLite default keeps local runs working with no configuration.
    """
    return os.getenv("MLFLOW_TRACKING_URI") or (
        "sqlite:///" + str(Path(__file__).resolve().parents[3] / "mlflow.db") + "?timeout=30"
    )


def _run_metric(client: MlflowClient, run_id: str, metric: str) -> float | None:
    """`metric` on a run, or None if the run is gone or never logged it."""
    try:
        return client.get_run(run_id).data.metrics.get(metric)
    except Exception:  # run deleted, or the tracking store can't be reached
        return None


def _incumbent(
        client: MlflowClient, registry_name: str, alias: str, metric: str
) -> tuple[int | None, float | None]:
    """(version, metric) behind `registry_name@alias`, or (None, None) if there isn't one yet."""
    try:
        version = client.get_model_version_by_alias(registry_name, alias)
    except Exception:  # first run for this model — no registered model, no alias
        return None, None
    return int(version.version), _run_metric(client, version.run_id, metric)


def register_champion(
        run_id: str,
        artifact_name: str,
        registry_name: str,
        metric: str = SELECTION_METRIC,
        alias: str = "champion",
) -> tuple[int, bool]:
    """
    Register a logged model under `registry_name` — but only if it beats the current champion.

    `alias` tracks the best version ever trained for this model, not the newest one. A run
    that scores worse is left out of the registry entirely; its MLflow run and model artifact
    still exist under the experiment, so nothing is lost, but `models:/<name>@<alias>` keeps
    resolving to the better model and downstream batch scoring never silently regresses.

    Args:
        run_id: MLflow run that logged the model.
        artifact_name: Model name used at log time (the `model-<name>` artifact path).
        registry_name: Registered model name to create/extend.
        metric: Run metric to rank by — higher wins. See `SELECTION_METRIC` on why this is a
            cross-validated score rather than a held-out one.
        alias: Alias to point at the champion.

    Returns:
        (champion version, whether this run was promoted). On a loss the version returned is
        the incumbent's, so callers always know what the alias resolves to.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    client = MlflowClient()

    challenger = _run_metric(client, run_id, metric)
    incumbent_version, incumbent = _incumbent(client, registry_name, alias, metric)

    # A tie keeps the incumbent, so re-running training on unchanged data is a no-op. An
    # unmeasurable challenger loses; an unmeasurable incumbent (its run was deleted) is
    # replaced by any challenger that does have a score.
    keep_incumbent = incumbent_version is not None and (
        challenger is None or (incumbent is not None and challenger <= incumbent)
    )
    if keep_incumbent:
        return incumbent_version, False

    version = mlflow.register_model(f"runs:/{run_id}/model-{artifact_name}", registry_name)
    client.set_registered_model_alias(registry_name, alias, version.version)
    return int(version.version), True


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

        # The search's own cross-validated score, alongside the held-out metrics: this is
        # what `register_champion` promotes on, so it has to live on the run. Every builder
        # returns a SearchCV, so `best_score_` is always there — the guard is for the day
        # one of them returns a plain estimator.
        if hasattr(model, "best_score_"):
            metrics[SELECTION_METRIC] = float(model.best_score_)

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
