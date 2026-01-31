import json
import os
from anyio import Path
from prefect import task


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