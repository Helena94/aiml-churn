import logging
from pathlib import Path

import pandas as pd
import yaml
from prefect import flow, task
from prefect.futures import wait

from churn.features.builder import preprocess_features, split_train_test
from churn.flows.mlflow import run_with_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

model_names = ["logistic_regression", "random_forest", "xgboost", "kmeans"]
config_paths = ["configs/logistic_regression.yml", "configs/random_forest.yml", "configs/xgboost.yml",
                "configs/kmeans.yml"]


@task(task_run_name="run-experiment-{model_name}")
def run_experiment(model_name: str, config, X_train, X_test, y_train, y_test):
    logger.info(f"Model Training: {model_name}")
    model, metrics = run_with_mlflow(X_train, X_test, y_train, y_test, model_name, config)
    logger.info(f"Results: {metrics}")
    return (model, metrics)


@flow(name="run_experiments")
def run_experiments():
    data_path = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = split_train_test(data)
    X_train, X_test = preprocess_features(X_train, X_test)
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


if __name__ == "__main__":
    run_experiments()
