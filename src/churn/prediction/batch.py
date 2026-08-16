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
from prefect.assets import materialize

from churn.assets import (
    BATCH_INPUT_ASSET,
    CLEANED_DATA_ASSET,
    COMPARISON_ASSET,
    PREDICTIONS_ASSET,
    add_metadata,
)
from churn.features.builder import build_features, split_train_test
from churn.features.schema import (
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)
from churn.flows.mlflow import mlflow_tracking_uri
from churn.training.predict import predict

logger = logging.getLogger(__name__)

CUSTOMER_ID = ID_COLUMNS[0]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
RISK_GROUPS = ("Low", "Medium", "High")
PROBABILITY_PREFIX = "churn_probability_"


# A materializing task declares its assets at import time, so these keys track the default
# paths in configs/batch_prediction.yml. `with_options` can override asset_deps at call time
# but not assets, so pointing the config elsewhere leaves these keys behind.
@materialize(BATCH_INPUT_ASSET, asset_deps=[CLEANED_DATA_ASSET])
def prepare_batch_input(input_path: str | Path, source_path: str | Path) -> bool:
    """
    Build the batch input from the held-out test rows when no external extract has landed.

    Stands in for an upstream customer extract. Only the test split is written, so the
    batch scores customers no model was fitted on. An existing batch is rebuilt once the
    cleaned data is newer than it, because every ETL run re-splits the holdout and a stale
    batch would score rows the models were just trained on. Set `fallback_source: null` in
    the config to turn the stand-in off entirely and protect a real extract.

    Args:
        input_path: Where the batch flow expects its input.
        source_path: Cleaned CSV to derive the batch from (the ETL output).

    Returns:
        True if a file was written, False if the existing input was current.
    """
    path = Path(input_path)
    source = Path(source_path)
    if path.exists() and (not source.exists() or path.stat().st_mtime >= source.stat().st_mtime):
        logger.info(f"Batch input at {path} is current, leaving it untouched")
        return False

    df = pd.read_csv(source_path)
    # ponytail: re-runs the same deterministic split as run_experiments.py instead of
    # persisting the holdout at training time — both call split_train_test() with its
    # defaults, so the rows match. Persist X_test from training if the params diverge.
    test_index = split_train_test(df)[1].index
    batch = df.loc[test_index].drop(columns=TARGET_COLUMN)

    path.parent.mkdir(parents=True, exist_ok=True)
    batch.to_parquet(path, index=False)
    logger.info(f"Built batch input at {path} ({len(batch)} held-out rows) from {source_path}")
    return True


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
    model contributes `segment` (plus `segment_distance` when it exposes
    `transform`); the primary model additionally drives the authoritative churn
    class, probability and risk group.

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
            # Distance to the assigned centroid, in the model's preprocessed space —
            # how typical a customer is of their segment. Guarded: a clustering model
            # without `transform` should cost the column, not the batch.
            if hasattr(model, "transform"):
                out["segment_distance"] = model.transform(X).min(axis=1)
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
def enrich_with_source(
    predictions: pd.DataFrame,
    source_path: str | Path | None,
) -> pd.DataFrame:
    """
    Attach each scored customer's cleaned-source row: features plus the true label.

    The scoring frame carries only `customer_id` and prediction columns, which is
    enough to act on but not to analyse. Merging the cleaned dataset back in gives
    a self-contained output: error metrics against `churn_actual`, and risk sliced
    by contract, tenure or charges — with no second file to join.

    The target is renamed `churn_actual` so it cannot be mistaken for `churn_label`,
    which is the *predicted* Yes/No.

    Args:
        predictions: Prediction DataFrame, one row per scored customer.
        source_path: Cleaned CSV to merge from, or None to skip enrichment (a real
            upstream extract has no labels to attach).

    Returns:
        The predictions with the source columns merged in, or unchanged if no
        source was given.
    """
    if not source_path:
        logger.info("No enrichment source configured; predictions keep the identifier only")
        return predictions

    source = pd.read_csv(source_path).rename(columns={TARGET_COLUMN[0]: "churn_actual"})
    # how="right": source columns land first, and the scored rows and their order win.
    enriched = source.merge(predictions, on=CUSTOMER_ID, how="right")

    missing = int(enriched["churn_actual"].isna().sum())
    if missing:
        logger.warning(f"{missing} scored customers had no matching row in {source_path}")

    logger.info(f"Enriched predictions with {len(source.columns) - 1} column(s) from {source_path}")
    return enriched


@materialize(PREDICTIONS_ASSET, asset_deps=[BATCH_INPUT_ASSET])
def save_predictions(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    batch_id: str,
) -> Path:
    """
    Write the timestamped historical file and the latest-predictions file.

    Both are written as parquet (the format downstream code reads) and as CSV
    (the format a human opens or plots from).

    Args:
        predictions: Prediction DataFrame with metadata.
        output_dir: Directory to write into (created if absent).
        batch_id: Identifier used in the historical filename.

    Returns:
        Path of the historical parquet file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    historical = out_dir / f"churn_predictions_{batch_id}.parquet"
    latest = out_dir / "latest_predictions.parquet"
    predictions.to_parquet(historical, index=False)
    predictions.to_parquet(latest, index=False)
    predictions.to_csv(historical.with_suffix(".csv"), index=False)
    predictions.to_csv(latest.with_suffix(".csv"), index=False)

    logger.info(f"Saved predictions to {historical} and {latest} (.parquet and .csv)")
    add_metadata(
        PREDICTIONS_ASSET,
        {
            "batch_id": batch_id,
            "rows": len(predictions),
            "predicted_churners": int(predictions["churn_prediction"].sum()),
        },
    )
    return historical


def _probability_columns(predictions: pd.DataFrame) -> list[str]:
    """Per-classifier probability columns, in output order."""
    return [c for c in predictions.columns if c.startswith(PROBABILITY_PREFIX)]


@task
def model_comparison(
    predictions: pd.DataFrame,
    model_uris: dict[str, str] | None = None,
    decision_threshold: float = 0.5,
    risk_low: float = 0.40,
    risk_high: float = 0.70,
) -> pd.DataFrame:
    """
    Build the multi-model comparison table from an already-scored batch.

    One row per classifier: its own probability distribution, churn count, risk
    split, registry lineage, and how often it agrees with the majority vote.
    Batch-level agreement statistics are repeated on every row as `batch_*`
    columns — constant across a 3-row table, which keeps the report to one file.

    Derives everything from the prediction columns, so no model is reloaded and
    nothing is re-scored.

    Args:
        predictions: Output of `generate_predictions` (before or after metadata).
        model_uris: Optional model name -> registry URI, for lineage columns.
        decision_threshold: Probability at or above which a model votes churn.
        risk_low: Lower risk boundary.
        risk_high: Upper risk boundary.

    Returns:
        Comparison DataFrame, one row per classifier.

    Raises:
        ValueError: If the frame carries no per-classifier probability columns.
    """
    columns = _probability_columns(predictions)
    if not columns:
        raise ValueError(
            f"No '{PROBABILITY_PREFIX}*' columns to compare; "
            "score the batch with at least one named classifier first"
        )

    uris = model_uris or {}
    probabilities = predictions[columns]
    votes = probabilities >= decision_threshold
    churn_votes = votes.sum(axis=1)
    n_models, n_rows = len(columns), len(predictions)

    # Majority vote. With an even number of models a tie counts as non-churn.
    consensus = churn_votes * 2 > n_models
    all_churn = int((churn_votes == n_models).sum())
    all_no_churn = int((churn_votes == 0).sum())
    batch_stats = {
        "batch_n_customers": n_rows,
        "batch_all_agree_churn": all_churn,
        "batch_all_agree_no_churn": all_no_churn,
        "batch_disagree": n_rows - all_churn - all_no_churn,
        "batch_agreement_rate": (all_churn + all_no_churn) / n_rows if n_rows else 0.0,
        "batch_mean_probability_spread": float(
            (probabilities.max(axis=1) - probabilities.min(axis=1)).mean()
        ),
    }

    rows = []
    for column in columns:
        name = column.removeprefix(PROBABILITY_PREFIX)
        probability = probabilities[column]
        groups = pd.Series(assign_risk_group(probability.to_numpy(), risk_low, risk_high))
        counts = groups.value_counts()
        rows.append(
            {
                "model": name,
                "model_uri": uris.get(name),
                "model_version": resolve_model_version(uris[name]) if name in uris else None,
                "mean_churn_probability": float(probability.mean()),
                "median_churn_probability": float(probability.median()),
                "n_predicted_churn": int(votes[column].sum()),
                "churn_rate": float(votes[column].mean()),
                **{f"risk_{group.lower()}": int(counts.get(group, 0)) for group in RISK_GROUPS},
                # Folds pairwise agreement into one number: how much of an
                # outlier this model is against the majority.
                "agreement_with_consensus": float((votes[column] == consensus).mean()),
                **batch_stats,
            }
        )

    comparison = pd.DataFrame(rows)
    logger.info(f"Compared {n_models} classifier(s) over {n_rows} customers:")
    for row in rows:
        logger.info(
            f"  {row['model']:<22} mean={row['mean_churn_probability']:.3f} "
            f"churn={row['n_predicted_churn']:<5} "
            f"L/M/H={row['risk_low']}/{row['risk_medium']}/{row['risk_high']} "
            f"consensus={row['agreement_with_consensus']:.3f}"
        )
    logger.info(
        f"  Agreement: {all_churn} all-churn, {all_no_churn} all-no-churn, "
        f"{batch_stats['batch_disagree']} split "
        f"(rate {batch_stats['batch_agreement_rate']:.3f}, "
        f"mean spread {batch_stats['batch_mean_probability_spread']:.3f})"
    )
    return comparison


@materialize(COMPARISON_ASSET, asset_deps=[PREDICTIONS_ASSET])
def save_comparison(
    comparison: pd.DataFrame,
    output_dir: str | Path,
    batch_id: str,
) -> Path:
    """
    Write the timestamped comparison report and the latest-comparison file.

    Mirrors `save_predictions`: one historical file per batch plus a fixed-name
    file for a future dashboard to read.

    Args:
        comparison: Comparison table from `model_comparison`.
        output_dir: Directory to write into (created if absent).
        batch_id: Identifier used in the historical filename.

    Returns:
        Path of the historical file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = comparison.copy()
    comparison.insert(0, "batch_id", batch_id)
    comparison["comparison_timestamp"] = datetime.now(timezone.utc).isoformat()

    historical = out_dir / f"model_comparison_{batch_id}.csv"
    latest = out_dir / "latest_model_comparison.csv"
    comparison.to_csv(historical, index=False)
    comparison.to_csv(latest, index=False)

    logger.info(f"Saved model comparison to {historical} and {latest}")
    add_metadata(COMPARISON_ASSET, {"batch_id": batch_id, "models_compared": len(comparison)})
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
        col.removeprefix(PROBABILITY_PREFIX): float(predictions[col].mean())
        for col in _probability_columns(predictions)
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
