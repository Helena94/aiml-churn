"""Prefect tasks for the telco churn batch prediction stage.

Scores a batch of customers with the registered MLflow models. Each model
answers a different analytical question, so every classifier contributes its own
probability column and the clustering model contributes a segment; the primary
model owns the authoritative class and risk band.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from prefect import task

from churn.features.builder import build_features
from churn.features.schema import CATEGORICAL_COLUMNS, ID_COLUMNS, NUMERICAL_COLUMNS
from churn.flows.mlflow import mlflow_tracking_uri
from churn.training.predict import predict

logger = logging.getLogger(__name__)

CUSTOMER_ID = ID_COLUMNS[0]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
RISK_GROUPS = ("Low", "Medium", "High")


@task
def load_batch_data(input_path: str | Path) -> pd.DataFrame:
    """
    Read the batch customer file.

    Args:
        input_path: Path to a .parquet or .csv file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Batch input file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format '{path.suffix}' (expected .parquet or .csv)")

    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


@task
def validate_batch_data(df: pd.DataFrame) -> None:
    """
    Validate the batch frame before any model touches it.

    Checks the dataset is non-empty, carries the identifier and every expected
    feature column, and has no duplicated customer IDs.

    Args:
        df: The loaded batch DataFrame.

    Raises:
        ValueError: If any check fails.
    """
    if df.empty:
        raise ValueError("Batch dataset is empty")

    if CUSTOMER_ID not in df.columns:
        raise ValueError(f"Missing identifier column: {CUSTOMER_ID}")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {missing}")

    duplicates = int(df[CUSTOMER_ID].duplicated().sum())
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicated {CUSTOMER_ID} value(s)")

    logger.info(f"Validation passed: {len(df)} rows, {len(FEATURE_COLUMNS)} feature columns")


@task
def load_mlflow_model(model_uri: str, tracking_uri: str | None = None) -> Any:
    """
    Load a registered model from MLflow. Never trains.

    Uses the sklearn flavour so the estimator keeps `predict_proba`, which the
    pyfunc wrapper does not expose.

    Args:
        model_uri: Registry URI, e.g. `models:/telco-churn-model@champion`.
        tracking_uri: Tracking/registry URI; defaults to the project SQLite store.

    Returns:
        The loaded sklearn estimator (preprocessing + model pipeline).
    """
    mlflow.set_tracking_uri(tracking_uri or mlflow_tracking_uri())
    logger.info(f"Loading model from {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def resolve_model_version(model_uri: str) -> str | None:
    """
    Best-effort lookup of the registry version behind a `models:/name@alias` URI.

    Args:
        model_uri: Model URI to resolve.

    Returns:
        The version string, or None if the URI is not an alias form or lookup fails.
    """
    if not model_uri.startswith("models:/") or "@" not in model_uri:
        return None
    name, _, alias = model_uri.removeprefix("models:/").partition("@")
    try:
        return str(MlflowClient().get_model_version_by_alias(name, alias).version)
    except Exception as exc:  # registry unreachable or alias gone — metadata only
        logger.warning(f"Could not resolve version for {model_uri}: {exc}")
        return None


def assign_risk_group(
    probabilities: np.ndarray,
    risk_low: float = 0.40,
    risk_high: float = 0.70,
) -> np.ndarray:
    """
    Bucket churn probabilities into risk groups.

    Low below `risk_low`, Medium in [`risk_low`, `risk_high`), High at or above
    `risk_high`.

    Args:
        probabilities: Churn probabilities.
        risk_low: Lower boundary (start of Medium).
        risk_high: Upper boundary (start of High).

    Returns:
        Array of "Low" / "Medium" / "High" labels.
    """
    probabilities = np.asarray(probabilities)
    return np.select(
        [probabilities >= risk_high, probabilities >= risk_low],
        ["High", "Medium"],
        default="Low",
    )


def _churn_probability(model: Any, X: pd.DataFrame, name: str) -> np.ndarray:
    """Positive-class churn probability for one classifier."""
    result = predict(model, X, return_proba=True)
    if "churn_probability" not in result:
        raise ValueError(
            f"Model '{name}' has no binary predict_proba output; "
            "if it is a clustering model, pass it as `segmentation`"
        )
    return result["churn_probability"]


@task
def generate_predictions(
    models: dict[str, Any],
    df: pd.DataFrame,
    primary: str,
    segmentation: str | None = None,
    decision_threshold: float = 0.5,
    risk_low: float = 0.40,
    risk_high: float = 0.70,
) -> pd.DataFrame:
    """
    Score the batch with every loaded model.

    Each classifier contributes `churn_probability_<name>`; the segmentation
    model contributes `segment`; the primary model additionally drives the
    authoritative churn class, probability and risk group.

    Args:
        models: Mapping of model name to loaded estimator.
        df: Validated batch DataFrame (raw columns, including customer_id).
        primary: Key in `models` whose output is authoritative.
        segmentation: Key in `models` to treat as a clustering model, if any.
        decision_threshold: Probability at or above which a customer is a churner.
        risk_low: Lower risk boundary.
        risk_high: Upper risk boundary.

    Returns:
        Prediction DataFrame, one row per input customer.

    Raises:
        ValueError: If the primary model was not loaded.
    """
    if primary not in models:
        raise ValueError(f"Primary model '{primary}' not available; loaded: {sorted(models)}")

    X = build_features(df)
    out = pd.DataFrame({CUSTOMER_ID: df[CUSTOMER_ID].to_numpy()})

    for name, model in models.items():
        if name == segmentation:
            out["segment"] = model.predict(X)
        else:
            out[f"churn_probability_{name}"] = _churn_probability(model, X, name)

    # `pop`: the primary's own column is superseded by `churn_probability`,
    # so it never appears twice in the output.
    probability = out.pop(f"churn_probability_{primary}").to_numpy()
    out["churn_probability"] = probability
    out["churn_prediction"] = (probability >= decision_threshold).astype(int)
    out["churn_label"] = np.where(out["churn_prediction"] == 1, "Yes", "No")
    out["risk_group"] = assign_risk_group(probability, risk_low, risk_high)

    logger.info(f"Scored {len(out)} customers with {len(models)} model(s), primary='{primary}'")
    return out


@task
def add_prediction_metadata(
    predictions: pd.DataFrame,
    model_uri: str,
    batch_id: str,
    model_version: str | None = None,
) -> pd.DataFrame:
    """
    Stamp lineage columns onto the predictions.

    Args:
        predictions: Prediction DataFrame.
        model_uri: URI of the primary model.
        batch_id: Identifier for this batch run.
        model_version: Resolved registry version, if known.

    Returns:
        The predictions with metadata columns added.
    """
    predictions = predictions.copy()
    predictions["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()
    predictions["model_uri"] = model_uri
    predictions["model_version"] = model_version
    predictions["batch_id"] = batch_id
    return predictions


@task
def save_predictions(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    batch_id: str,
) -> Path:
    """
    Write the timestamped historical file and the latest-predictions file.

    Args:
        predictions: Prediction DataFrame with metadata.
        output_dir: Directory to write into (created if absent).
        batch_id: Identifier used in the historical filename.

    Returns:
        Path of the historical file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    historical = out_dir / f"churn_predictions_{batch_id}.parquet"
    latest = out_dir / "latest_predictions.parquet"
    predictions.to_parquet(historical, index=False)
    predictions.to_parquet(latest, index=False)

    logger.info(f"Saved predictions to {historical} and {latest}")
    return historical


@task
def log_batch_summary(predictions: pd.DataFrame, output_path: str | Path) -> dict:
    """
    Log and return a summary of the batch run.

    Args:
        predictions: Final prediction DataFrame.
        output_path: Where the historical file was written.

    Returns:
        Summary dictionary.
    """
    counts = predictions["risk_group"].value_counts()
    summary = {
        "n_customers": len(predictions),
        "n_predicted_churn": int(predictions["churn_prediction"].sum()),
        "risk_groups": {group: int(counts.get(group, 0)) for group in RISK_GROUPS},
        "avg_churn_probability": float(predictions["churn_probability"].mean()),
        "output_path": str(output_path),
    }

    # Side-by-side models are the point of this batch, so surface their spread too.
    per_model = {
        col.removeprefix("churn_probability_"): float(predictions[col].mean())
        for col in predictions.columns
        if col.startswith("churn_probability_")
    }
    if per_model:
        summary["avg_churn_probability_per_model"] = per_model
    if "segment" in predictions.columns:
        summary["segments"] = {
            str(k): int(v) for k, v in predictions["segment"].value_counts().items()
        }

    logger.info("Batch prediction summary:")
    logger.info(f"  Customers processed:   {summary['n_customers']}")
    logger.info(f"  Predicted to churn:    {summary['n_predicted_churn']}")
    logger.info(f"  Risk groups:           {summary['risk_groups']}")
    logger.info(f"  Avg churn probability: {summary['avg_churn_probability']:.4f}")
    if per_model:
        logger.info(f"  Avg per model:         {per_model}")
    if "segments" in summary:
        logger.info(f"  Segments:              {summary['segments']}")
    logger.info(f"  Output:                {summary['output_path']}")

    return summary
