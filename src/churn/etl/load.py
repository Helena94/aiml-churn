import json
import pandas as pd
from prefect import task


@task(name="load_data")
def load_data(df: pd.DataFrame, transformation_log: dict, output_dir: str | None = None) -> None:
    """
    Load the transformed data and schema information to disk.

    Args:
        df: Transformed DataFrame to save
        transformation_log: Transformation log dictionary
        output_dir: Directory to save outputs (default: data/processed/)
    Returns:
        None
    """
    
    output_dir = output_dir / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "churn_cleaned.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    log_path = output_dir / "transformation_log.json"
    with open(log_path, "w") as f:
        json.dump(transformation_log, f, indent=2)
    print(f"  Saved: {log_path}")