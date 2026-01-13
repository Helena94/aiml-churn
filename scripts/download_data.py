"""Script to download dataset from Kaggle."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import download_dataset


def main():
    """Download telco customer churn dataset."""
    DATASET = "blastchar/telco-customer-churn"
    download_dataset(DATASET)


if __name__ == "__main__":
    main()
