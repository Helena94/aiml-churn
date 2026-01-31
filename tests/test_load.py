"""Unit tests for ETL load functions."""

import json
from pathlib import Path

import pandas as pd
import pytest

from churn.etl.load import load_data


class TestLoadData:
    """Tests for load_data function."""

    def test_saves_csv_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        transformation_log = {"steps": {}}
        load_data.fn(df, transformation_log, output_dir=tmp_path)
        csv_path = tmp_path / "processed" / "churn_cleaned.csv"
        assert csv_path.exists()

    def test_saves_transformation_log(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        transformation_log = {"initial_shape": {"rows": 2, "columns": 1}}
        load_data.fn(df, transformation_log, output_dir=tmp_path)
        log_path = tmp_path / "processed" / "transformation_log.json"
        assert log_path.exists()
        with open(log_path) as f:
            result = json.load(f)
        assert result["initial_shape"]["rows"] == 2

    def test_creates_output_directory(self, tmp_path):
        output_dir = tmp_path / "new_output"
        df = pd.DataFrame({"a": [1]})
        transformation_log = {}
        load_data.fn(df, transformation_log, output_dir=output_dir)
        assert (output_dir / "processed").exists()

    def test_csv_content_matches_dataframe(self, tmp_path):
        df = pd.DataFrame({"customer_id": ["A", "B"], "tenure": [10, 20]})
        transformation_log = {}
        load_data.fn(df, transformation_log, output_dir=tmp_path)
        csv_path = tmp_path / "processed" / "churn_cleaned.csv"
        loaded_df = pd.read_csv(csv_path)
        assert loaded_df["customer_id"].tolist() == ["A", "B"]
        assert loaded_df["tenure"].tolist() == [10, 20]

    def test_saves_without_index(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        transformation_log = {}
        load_data.fn(df, transformation_log, output_dir=tmp_path)
        csv_path = tmp_path / "processed" / "churn_cleaned.csv"
        loaded_df = pd.read_csv(csv_path)
        # Should only have column 'a', not an index column
        assert list(loaded_df.columns) == ["a"]
