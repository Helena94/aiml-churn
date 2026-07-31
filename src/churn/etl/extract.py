"""Extract raw data from Kaggle and record metadata for traceability."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from prefect import task
from prefect.assets import materialize
from prefect.logging import get_run_logger

from churn.assets import EXTRACTION_LOG_ASSET, KAGGLE_SOURCE_ASSET

from .download import download_dataset


def read_raw_csv(path) -> pd.DataFrame:
    """
    Read a raw CSV file into a DataFrame.

    Args:
        path: Path to the raw CSV file
    Returns:
        pd.DataFrame: Loaded DataFrame
    """

    churn_dt = pd.read_csv(path)
    return churn_dt


def compute_dataset_hash(df) -> str:
    """
    Compute an MD5 hash of the DataFrame for integrity verification.

    Args:
        df: pd.DataFrame to hash
    Returns:
        str: MD5 hash of the DataFrame
    """
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


@materialize(EXTRACTION_LOG_ASSET, asset_deps=[KAGGLE_SOURCE_ASSET])
def write_extraction_metadata(data_path, metadata):
    """
    Write extraction metadata to a JSON file.

    Args:
        data_path: Path to save the metadata JSON file
        metadata: dict containing metadata information
    Returns:
        None
    """
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w") as f:
        json.dump(metadata, f, indent=2)


@task(name="extract_data", retries=3)
def extract_data(data_path: Path, dataset_name: str, logger=None) -> pd.DataFrame:
    """
    Extract telco customer churn data from Kaggle.

    Downloads the dataset if not present, loads it into a DataFrame,
    and records metadata for future reference and reproducibility.

    Args:
        data_path: Base path for data storage (raw data will be saved under data_path)
        dataset_name: Kaggle dataset identifier (e.g., 'blastchar/telco-customer-churn')
        logger: Optional Prefect logger for logging messages

    Returns:
        tuple containing:
            - pd.DataFrame: The raw churn dataset
    """
    data_path_raw = data_path / "raw"

    # Download and load data
    DATA_PATH = download_dataset(data_path_raw, dataset_name, unzip=True)
    if logger:
        logger.info(f"Data downloaded to: {DATA_PATH}")

    if logger:
        logger.info("Reading raw CSV data into DataFrame...")
    churn_dt = read_raw_csv(DATA_PATH)

    # Record metadata for traceability
    if logger:
        logger.info("Recording extraction metadata...")
    data_hash = compute_dataset_hash(churn_dt)
    metadata = {
        "source": {
            "name": "Telco Customer Churn",
            "origin": "Kaggle",
            "dataset_id": dataset_name,
            "extracted_at": datetime.now().isoformat(),
        },
        "raw_data": {
            "file_path": str(DATA_PATH),
            "rows": len(churn_dt),
            "columns": len(churn_dt.columns),
            "column_names": list(churn_dt.columns),
            "dtypes": {col: str(dtype) for col, dtype in churn_dt.dtypes.items()},
            "memory_usage_mb": round(
                churn_dt.memory_usage(deep=True).sum() / 1024 / 1024, 2
            ),
            "hash": data_hash,
        },
    }

    # Save metadata to JSON file
    metadata_path = data_path / "raw" / "extraction_metadata_log.json"
    write_extraction_metadata(metadata_path, metadata)

    if logger:
        logger.info(f"Extraction metadata saved to: {metadata_path}")
    return churn_dt, data_hash
