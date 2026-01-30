"""Entry point for running the ETL pipeline."""
import json
import os
from pathlib import Path
from pdb import main

from prefect import flow, task

from churn.etl.extract import extract_data
from churn.etl.load import load_data
from churn.etl.transform import transform_data

PROJECT_ROOT = Path(__file__).parent.parent


@task(name="load_kaggle_credentials")
def load_kaggle_credentials(project_root: Path):
    """Load Kaggle credentials from project config and set as env vars."""
    kaggle_config = project_root / "kaggle" / "kaggle.json"

    if kaggle_config.exists():
        with open(kaggle_config) as f:
            creds = json.load(f)
            os.environ["KAGGLE_USERNAME"] = creds["username"]
            os.environ["KAGGLE_KEY"] = creds["key"]
    else:
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_config}. "
            "Please create kaggle/kaggle.json with your credentials."
        )

@flow(name="etl_pipeline", timeout_seconds=30, log_prints=True)
def run_etl():
    """Run the ETL pipeline."""
    data_path = PROJECT_ROOT / "data"
    load_kaggle_credentials(PROJECT_ROOT)

    data = extract_data(data_path, "blastchar/telco-customer-churn")
    transformed_data, transformation_log = transform_data(data)
    load_data(transformed_data, transformation_log, output_dir=data_path)


if __name__ == "__main__":
    run_etl()
