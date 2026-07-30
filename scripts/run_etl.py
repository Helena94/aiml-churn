"""Entry point for running the ETL pipeline."""
from datetime import datetime
from pathlib import Path

from prefect import flow

from churn.etl import load_kaggle_credentials
from churn.etl.extract import extract_data
from churn.etl.load import load_data
from churn.etl.transform import transform_data
from prefect.logging import get_run_logger

PROJECT_ROOT = Path(__file__).parent.parent

def generate_flow_run_name() -> str:
    return f"etl_pipeline-{datetime.now():%A}"


@flow(flow_run_name=generate_flow_run_name, timeout_seconds=900, log_prints=True)
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
