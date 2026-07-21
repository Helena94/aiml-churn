import logging
from pathlib import Path

import pandas as pd
import yaml
from prefect import flow, task
from prefect.futures import wait

from churn.features.builder import split_train_test
from churn.flows.mlflow import register_champion, run_with_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

model_names = ["logistic_regression", "random_forest", "xgboost", "kmeans"]
config_paths = ["configs/logistic_regression.yml", "configs/random_forest.yml", "configs/xgboost.yml",
                "configs/kmeans.yml"]

# Each model answers a different analytical question, so each gets its own
# registered name rather than competing for a single slot.
REGISTRY_NAMES = {
    "logistic_regression": "telco-churn-logistic-regression",
    "random_forest": "telco-churn-random-forest",
    "xgboost": "telco-churn-xgboost",
    "kmeans": "telco-churn-segmentation",
}
CLASSIFIERS = ("logistic_regression", "random_forest", "xgboost")
PRIMARY_REGISTRY_NAME = "telco-churn-model"
PRIMARY_METRIC = "roc_auc"


@task(task_run_name="run-experiment-{model_name}")
def run_experiment(model_name: str, config, X_train, X_test, y_train, y_test):
    logger.info(f"Model Training: {model_name}")
    model, metrics, run_id = run_with_mlflow(X_train, X_test, y_train, y_test, model_name, config)
    logger.info(f"Results: {metrics}")
    return (model_name, metrics, run_id)


@task
def register_models(results: list[tuple[str, dict, str]]) -> None:
    """Register every trained model under its own name, then alias the best classifier as primary."""
    for model_name, _, run_id in results:
        version = register_champion(run_id, model_name, REGISTRY_NAMES[model_name])
        logger.info(f"Registered {REGISTRY_NAMES[model_name]} v{version} (champion)")

    # kmeans is clustering — it has no roc_auc and never competes for primary.
    ranked = [(m, mx) for m, mx, _ in results if m in CLASSIFIERS and PRIMARY_METRIC in mx]
    if not ranked:
        logger.warning(f"No classifier reported {PRIMARY_METRIC}; skipping {PRIMARY_REGISTRY_NAME}")
        return

    best_name = max(ranked, key=lambda r: r[1][PRIMARY_METRIC])[0]
    best_run_id = next(r for m, _, r in results if m == best_name)
    version = register_champion(best_run_id, best_name, PRIMARY_REGISTRY_NAME)
    logger.info(f"Primary champion: {best_name} -> {PRIMARY_REGISTRY_NAME} v{version}")


@flow(name="run_experiments")
def run_experiments():
    data_path = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    # Raw frames go straight into the pipelines — preprocessing now lives
    # inside each model, so the logged artifact can score raw data.
    X_train, X_test, y_train, y_test = split_train_test(data)
    # Encode target: Yes -> 1, No -> 0
    y_train = y_train.map({"Yes": 1, "No": 0})
    y_test = y_test.map({"Yes": 1, "No": 0})
    futures = []
    for model_name, config_path in zip(model_names, config_paths):
        logger.info(f"Loading config from {config_path}")
        with open(PROJECT_ROOT / config_path) as f:
            config = yaml.safe_load(f)
        future = run_experiment.submit(model_name, config, X_train, X_test, y_train, y_test)
        futures.append(future)
    wait(futures)

    register_models([f.result() for f in futures])


if __name__ == "__main__":
    run_experiments()
