import argparse
import logging
from pathlib import Path

import yaml
import pandas as pd

from churn.features.builder import split_train_test
from churn.flows.mlflow import run_with_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def run_experiment(model_name: str, config_path: str):
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_path = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = split_train_test(data)

    # Encode target: Yes -> 1, No -> 0
    y_train = y_train.map({"Yes": 1, "No": 0})
    y_test = y_test.map({"Yes": 1, "No": 0})

    model, metrics = run_with_mlflow(X_train, X_test, y_train, y_test, model_name, config)
    logger.info(f"Results: {metrics}")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a churn model experiment with MLflow tracking")
    parser.add_argument(
        "--model",
        required=True,
        choices=["logistic_regression", "random_forest", "xgboost", "kmeans"],
        help="Model to train",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. configs/logistic_regression.yml)",
    )
    args = parser.parse_args()

    run_experiment(args.model, args.config)
