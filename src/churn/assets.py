"""Asset keys and metadata helper for the Prefect Cloud lineage graph.

Keys live in one module for the same reason column names live in `features/schema.py`:
they are referenced from the ETL, training and prediction stages, and a typo doesn't fail —
it silently splits one node in the graph into two.
"""

from typing import Any

from prefect.assets import add_asset_metadata as _add_asset_metadata
from prefect.context import AssetContext

# Kaggle is read, never written by this pipeline — declared as a dependency so the graph
# starts at the real origin instead of mid-stream.
KAGGLE_SOURCE_ASSET = "kaggle://blastchar/telco-customer-churn"
EXTRACTION_LOG_ASSET = "file://data/raw/extraction_metadata_log.json"
TRANSFORMATION_LOG_ASSET = "file://data/processed/transformation_log.json"
CLEANED_DATA_ASSET = "file://data/processed/churn_cleaned.csv"
BATCH_INPUT_ASSET = "file://data/processed/batch_customers.parquet"
PREDICTIONS_ASSET = "file://data/predictions/latest_predictions.parquet"
COMPARISON_ASSET = "file://data/predictions/latest_model_comparison.csv"


def model_asset(registry_name_or_uri: str) -> str:
    """Asset key for a registered MLflow model, from a registry name or a `models:/` URI.

    `telco-churn-model` and `models:/telco-churn-model@champion` both map to
    `mlflow://telco-churn-model`, so training and scoring land on the same graph node.
    Handles the `@alias` and `/version` URI forms.
    """
    name = registry_name_or_uri.removeprefix("models:/").split("@")[0].split("/")[0]
    return f"mlflow://{name}"


def add_metadata(asset: str, metadata: dict[str, Any]) -> None:
    """`add_asset_metadata` that no-ops outside a materializing task run.

    Prefect raises without an AssetContext, which is the case whenever a task's `.fn()` is
    called directly — this project's convention for bypassing the Prefect runtime in tests.
    """
    if AssetContext.get():
        _add_asset_metadata(asset, metadata)
