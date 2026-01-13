"""Download datasets from Kaggle to raw_data folder."""
import json
import os
from pathlib import Path

# Load Kaggle credentials from project's kaggle folder and set as env vars
_project_root = Path(__file__).parent.parent.parent
_kaggle_config = _project_root / "kaggle" / "kaggle.json"
if _kaggle_config.exists():
    with open(_kaggle_config) as f:
        _creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = _creds["username"]
        os.environ["KAGGLE_KEY"] = _creds["key"]

from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(dataset_name: str, unzip: bool = True) -> Path:
    """
    Download a dataset from Kaggle.

    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'blastchar/telco-customer-churn')
        unzip: Whether to unzip the downloaded files

    Returns:
        Path to the downloaded CSV file
    """
    # Get project root and raw path
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / "data" / "raw"
    raw_data_path.mkdir(parents=True, exist_ok=True)

    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    print(f"Downloading {dataset_name} to {raw_data_path}...")
    api.dataset_download_files(dataset_name, path=str(raw_data_path), unzip=unzip)
    print("Download complete!")

    # Find and return the CSV file path
    csv_files = list(raw_data_path.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    return raw_data_path

