import json
import os
from anyio import Path
from prefect import task


@task(name="load_kaggle_credentials", retries=3, retry_delay_seconds=[2, 4, 8])
def load_kaggle_credentials(project_root: Path, logger=None) -> None:
    """Ensure KAGGLE_USERNAME / KAGGLE_KEY are set, from the environment or kaggle/kaggle.json.

    Already-set env vars win, so a deployment can supply them as secrets without the file.

    Args:
        project_root: Path to the project root directory
        logger: Optional Prefect logger for logging messages
    Returns:
        None"""
    # Env vars are the contract download.py actually relies on; the JSON file is just the
    # local way to populate them. On Prefect Cloud they arrive as deployment secrets and
    # there is no kaggle.json in the clone.
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        if logger:
            logger.info("Kaggle credentials found in environment.")
        return

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
            "Create kaggle/kaggle.json with your credentials, "
            "or set KAGGLE_USERNAME and KAGGLE_KEY in the environment."
        )