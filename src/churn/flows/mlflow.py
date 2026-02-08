import mlflow.sklearn
import pandas as pd

import mlflow
from churn.training.train import train_model


def run_one_model(X: pd.DataFrame, Y: pd.DataFrame, model_name: str, config: dict):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("churn-training")

    with mlflow.start_run(run_name=model_name):
        result = train_model(X, Y, model_name=model_name, config=config)

        mlflow.log_params(result["best_params"])
        mlflow.log_metrics(result["metrics"])
        mlflow.log_artifact(result["metrics_path"])
        mlflow.log_artifact(result["model_path"])  # OR log_model if you prefer

    return result
