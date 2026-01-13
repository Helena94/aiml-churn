"""Extract raw data from Kaggle and record metadata for traceability."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from etl.download import download_dataset


def extract() -> tuple[pd.DataFrame, dict]:
    """
    Extract telco customer churn data from Kaggle.

    Downloads the dataset if not present, loads it into a DataFrame,
    and records metadata for future reference and reproducibility.

    Returns:
        tuple containing:
            - pd.DataFrame: The raw churn dataset
            - dict: Metadata about the extraction (source, timestamp, shape, etc.)
    """
    SRC_DIR = Path.cwd().parent.parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    print(f"Added to sys.path: {SRC_DIR}")

    # Download and load data
    dataset_name = "blastchar/telco-customer-churn"
    DATA_PATH = download_dataset(dataset_name, unzip=True)
    print(f"Loading: {DATA_PATH}")
    churn_dt = pd.read_csv(DATA_PATH)

    # Record metadata for traceability
    data_hash = hashlib.md5(pd.util.hash_pandas_object(churn_dt).values).hexdigest()
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
            "memory_usage_mb": round(churn_dt.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "hash": data_hash,
        },
    }

    # Save metadata to JSON file
    project_root = Path(__file__).parent.parent.parent
    metadata_path = project_root / "data" / "raw" / "extraction_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return churn_dt, metadata

