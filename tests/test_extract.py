"""Unit tests for ETL extract functions."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from churn.etl.extract import (
    compute_dataset_hash,
    read_raw_csv,
    write_extraction_metadata,
)


class TestReadRawCsv:
    """Tests for read_raw_csv function."""

    def test_reads_csv_file(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n")
        result = read_raw_csv(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_preserves_data_types(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,value\nfoo,100\nbar,200\n")
        result = read_raw_csv(csv_path)
        assert result["value"].dtype in ["int64", "float64"]


class TestComputeDatasetHash:
    """Tests for compute_dataset_hash function."""

    def test_returns_md5_hash(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = compute_dataset_hash(df)
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hash length

    def test_same_data_same_hash(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert compute_dataset_hash(df1) == compute_dataset_hash(df2)

    def test_different_data_different_hash(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        assert compute_dataset_hash(df1) != compute_dataset_hash(df2)


class TestWriteExtractionMetadata:
    """Tests for write_extraction_metadata function."""

    def test_writes_json_file(self, tmp_path):
        metadata_path = tmp_path / "metadata.json"
        metadata = {"source": "test", "rows": 100}
        write_extraction_metadata.fn(metadata_path, metadata)
        assert metadata_path.exists()
        with open(metadata_path) as f:
            result = json.load(f)
        assert result == metadata

    def test_creates_parent_directories(self, tmp_path):
        metadata_path = tmp_path / "nested" / "dir" / "metadata.json"
        metadata = {"test": "data"}
        write_extraction_metadata.fn(metadata_path, metadata)
        assert metadata_path.exists()

    def test_writes_formatted_json(self, tmp_path):
        metadata_path = tmp_path / "metadata.json"
        metadata = {"a": 1, "b": 2}
        write_extraction_metadata.fn(metadata_path, metadata)
        content = metadata_path.read_text()
        assert "  " in content  # Check for indentation
