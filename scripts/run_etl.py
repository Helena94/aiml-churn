"""Entry point for running the ETL pipeline."""

import json
import os
from datetime import datetime
from pathlib import Path

from prefect import flow, task

from churn.etl.extract import extract_data
from churn.etl.load import load_data
from churn.etl.transform import transform_data
from prefect.logging import get_run_logger

PROJECT_ROOT = Path(__file__).parent.parent


@task(name="load_kaggle_credentials", retries=3, retry_delay_seconds=[2, 4, 8])
def load_kaggle_credentials(project_root: Path, logger=None) -> None:
    """Load Kaggle credentials from project config and set as env vars.
    Args:
        project_root: Path to the project root directory
        logger: Optional Prefect logger for logging messages
    Returns:
        None"""
    kaggle_config = project_root / "kaggle" / "kaggle.json"

    if kaggle_config.exists():
        with open(kaggle_config) as f:
            creds = json.load(f)
            os.environ["KAGGLE_USERNAME"] = creds["username"]
            os.environ["KAGGLE_KEY"] = creds["key"]
        if logger:
            logger.info("Kaggle credentials loaded successfully.")
    else:
        if logger:
            logger.error("Kaggle credentials file not found.")
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_config}. "
            "Please create kaggle/kaggle.json with your credentials."
        )


def generate_flow_run_name() -> str:
    return f"etl_pipeline-{datetime.now():%A}"


@flow(flow_run_name=generate_flow_run_name, timeout_seconds=30, log_prints=True)
def run_etl():
    """Run the ETL pipeline."""
    logger = get_run_logger()
    logger.info("Starting ETL pipeline...")

    data_path = PROJECT_ROOT / "data"
    logger.info("Loading Kaggle credentials...")
    load_kaggle_credentials(PROJECT_ROOT, logger=logger)
    logger.info(f"Data path set to: {data_path}")

    logger.info("Extracting data...")
    data, data_hash = extract_data(data_path, "blastchar/telco-customer-churn", logger=logger)
    logger.info("Transforming data...")
    transformed_data, transformation_log = transform_data(data, data_hash, logger=logger)
    logger.info("Loading data...")
    load_data(transformed_data, transformation_log, output_dir=data_path, logger=logger)
    logger.info("ETL pipeline completed successfully.")


if __name__ == "__main__":
    run_etl()
