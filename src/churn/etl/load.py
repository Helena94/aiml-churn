import json
import pandas as pd
from prefect.assets import materialize

from churn.assets import (
    CLEANED_DATA_ASSET,
    EXTRACTION_LOG_ASSET,
    TRANSFORMATION_LOG_ASSET,
    add_metadata,
)


# `materialize` is itself a task decorator — stacking @task on top would wrap the
# materializing task and lose the asset, so this is the only decorator here.
@materialize(
    TRANSFORMATION_LOG_ASSET,
    CLEANED_DATA_ASSET,
    asset_deps=[EXTRACTION_LOG_ASSET],
)
def load_data(
    df: pd.DataFrame,
    transformation_log: dict,
    output_dir: str | None = None,
    logger=None,
) -> None:
    """
    Load the transformed data and schema information to disk.

    Args:
        df: Transformed DataFrame to save
        transformation_log: Transformation log dictionary
        output_dir: Directory to save outputs (default: data/processed/)
        logger: Optional Prefect logger for logging progress
    Returns:
        None
    """

    if logger:
        logger.info("Loading transformed data to disk...")
    output_dir = output_dir / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "churn_cleaned.csv"
    df.to_csv(csv_path, index=False)
    if logger:
        logger.info(f"  Saved: {csv_path}")

    if logger:
        logger.info("Saving transformation log...")
    log_path = output_dir / "transformation_log.json"
    with open(log_path, "w") as f:
        json.dump(transformation_log, f, indent=2)
    if logger:
        logger.info(f"  Saved: {log_path}")

    add_metadata(CLEANED_DATA_ASSET, {"rows": len(df), "columns": len(df.columns)})
