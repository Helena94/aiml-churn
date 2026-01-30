"""Download datasets from Kaggle to raw_data folder.

Note: Kaggle credentials (KAGGLE_USERNAME, KAGGLE_KEY) must be set
as environment variables before calling download functions.
"""

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from prefect import task


def download_dataset(path_data: Path, dataset_name: str, unzip: bool = True) -> Path:
    """
    Download a dataset from Kaggle.

    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'blastchar/telco-customer-churn')
        unzip: Whether to unzip the downloaded files

    Returns:
        Path to the downloaded CSV file
    """

    path_data.mkdir(parents=True, exist_ok=True)

    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    print(f"Downloading {dataset_name} to {path_data}...")
    api.dataset_download_files(dataset_name, path=str(path_data), unzip=unzip)
    print("Download complete!")

    # Find and return the CSV file path
    csv_files = list(path_data.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    return path_data
