import mlflow
import mlflow.sklearn
import pandas as pd

from churn.training.train import train_model
from churn.training.evaluation import evaluate_model


def run_with_mlflow(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    config: dict,
):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("churn-training")

    with mlflow.start_run(run_name=model_name):
        model = train_model(X_train, y_train, model_name=model_name, config=config)

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
        mlflow.sklearn.log_model(model, artifact_path="model")

    return model, metrics
