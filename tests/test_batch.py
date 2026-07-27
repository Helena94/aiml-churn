"""Unit tests for the batch prediction tasks.

Prefect tasks are exercised through `.fn()` to bypass the Prefect runtime.
"""

import numpy as np
import pandas as pd
import pytest

from churn.features.builder import split_train_test
from churn.prediction.batch import (
    FEATURE_COLUMNS,
    assign_risk_group,
    generate_predictions,
    load_batch_data,
    log_batch_summary,
    prepare_batch_input,
    save_predictions,
    validate_batch_data,
)


def _batch_df(n: int = 4) -> pd.DataFrame:
    """Minimal valid batch frame: identifier plus every expected feature column."""
    df = pd.DataFrame({"customer_id": [f"id-{i}" for i in range(n)]})
    for col in FEATURE_COLUMNS:
        df[col] = list(range(n))
    return df


class StubClassifier:
    """Fake classifier exposing binary predict_proba."""

    def __init__(self, churn_proba):
        self._proba = np.column_stack([1 - np.asarray(churn_proba), np.asarray(churn_proba)])

    def predict(self, X):
        return (self._proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        return self._proba


class StubSegmenter:
    """Fake clustering model — labels only, no predict_proba."""

    def __init__(self, labels):
        self._labels = np.asarray(labels)

    def predict(self, X):
        return self._labels


class TestPrepareBatchInput:
    """Tests for prepare_batch_input task."""

    def _source(self, tmp_path, n=100):
        """Cleaned-CSV stand-in: features plus the identifier and target columns."""
        df = _batch_df(n)
        df["churn"] = ["Yes" if i % 4 == 0 else "No" for i in range(n)]
        path = tmp_path / "cleaned.csv"
        df.to_csv(path, index=False)
        return path

    def test_writes_only_the_holdout_rows(self, tmp_path):
        source = self._source(tmp_path, n=100)
        out = tmp_path / "batch.parquet"

        assert prepare_batch_input.fn(out, source) is True

        written = pd.read_parquet(out)
        assert len(written) == 20  # 20% test split
        assert "churn" not in written.columns
        assert "customer_id" in written.columns

    def test_rows_match_the_training_holdout(self, tmp_path):
        """The batch must be the same rows run_experiments.py held out."""
        source = self._source(tmp_path, n=100)
        out = tmp_path / "batch.parquet"
        prepare_batch_input.fn(out, source)

        df = pd.read_csv(source)
        expected = set(df.loc[split_train_test(df)[1].index, "customer_id"])
        assert set(pd.read_parquet(out)["customer_id"]) == expected

    def test_existing_file_is_not_overwritten(self, tmp_path):
        source = self._source(tmp_path)
        out = tmp_path / "batch.parquet"
        _batch_df(3).to_parquet(out, index=False)

        assert prepare_batch_input.fn(out, source) is False
        assert len(pd.read_parquet(out)) == 3


class TestLoadBatchData:
    """Tests for load_batch_data task."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_batch_data.fn(tmp_path / "nope.parquet")

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "batch.txt"
        path.write_text("nope")

        with pytest.raises(ValueError, match="Unsupported input format"):
            load_batch_data.fn(path)

    def test_reads_parquet_and_csv(self, tmp_path):
        df = _batch_df(2)
        df.to_parquet(tmp_path / "batch.parquet", index=False)
        df.to_csv(tmp_path / "batch.csv", index=False)

        assert len(load_batch_data.fn(tmp_path / "batch.parquet")) == 2
        assert len(load_batch_data.fn(tmp_path / "batch.csv")) == 2


class TestValidateBatchData:
    """Tests for validate_batch_data task."""

    def test_valid_frame_passes(self):
        validate_batch_data.fn(_batch_df())

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_batch_data.fn(_batch_df().iloc[:0])

    def test_missing_identifier_raises(self):
        with pytest.raises(ValueError, match="Missing identifier column"):
            validate_batch_data.fn(_batch_df().drop(columns=["customer_id"]))

    def test_missing_feature_column_raises(self):
        df = _batch_df().drop(columns=[FEATURE_COLUMNS[0]])

        with pytest.raises(ValueError, match="Missing required feature column"):
            validate_batch_data.fn(df)

    def test_duplicate_customer_ids_raise(self):
        df = _batch_df()
        df.loc[1, "customer_id"] = df.loc[0, "customer_id"]

        with pytest.raises(ValueError, match="duplicated customer_id"):
            validate_batch_data.fn(df)


class TestAssignRiskGroup:
    """Tests for the risk banding helper."""

    def test_boundaries(self):
        groups = assign_risk_group(np.array([0.0, 0.39, 0.40, 0.69, 0.70, 1.0]))

        assert list(groups) == ["Low", "Low", "Medium", "Medium", "High", "High"]

    def test_custom_thresholds(self):
        groups = assign_risk_group(np.array([0.15, 0.25, 0.55]), risk_low=0.2, risk_high=0.5)

        assert list(groups) == ["Low", "Medium", "High"]


class TestGeneratePredictions:
    """Tests for generate_predictions task."""

    def _models(self):
        return {
            "primary": StubClassifier([0.1, 0.5, 0.8, 0.95]),
            "random_forest": StubClassifier([0.2, 0.4, 0.6, 0.9]),
            "segment": StubSegmenter([0, 1, 1, 0]),
        }

    def _run(self):
        return generate_predictions.fn(
            self._models(), _batch_df(), primary="primary", segmentation="segment"
        )

    def test_primary_drives_class_and_risk(self):
        out = self._run()

        assert out["churn_probability"].tolist() == [0.1, 0.5, 0.8, 0.95]
        assert out["churn_prediction"].tolist() == [0, 1, 1, 1]
        assert out["churn_label"].tolist() == ["No", "Yes", "Yes", "Yes"]
        assert out["risk_group"].tolist() == ["Low", "Medium", "High", "High"]

    def test_keeps_comparison_and_segment_columns(self):
        out = self._run()

        assert out["churn_probability_random_forest"].tolist() == [0.2, 0.4, 0.6, 0.9]
        assert out["segment"].tolist() == [0, 1, 1, 0]

    def test_primary_column_not_duplicated(self):
        # `churn_probability` supersedes the primary's own prefixed column.
        assert "churn_probability_primary" not in self._run().columns

    def test_customer_id_preserved(self):
        out = self._run()

        assert out["customer_id"].tolist() == ["id-0", "id-1", "id-2", "id-3"]
        assert len(out) == 4

    def test_decision_threshold_is_honoured(self):
        out = generate_predictions.fn(
            self._models(),
            _batch_df(),
            primary="primary",
            segmentation="segment",
            decision_threshold=0.85,
        )

        assert out["churn_prediction"].tolist() == [0, 0, 0, 1]

    def test_undeclared_clustering_model_raises(self):
        # Forgetting `segmentation` must fail loudly and name the model.
        with pytest.raises(ValueError, match="Model 'segment' has no binary predict_proba"):
            generate_predictions.fn(self._models(), _batch_df(), primary="primary")

    def test_missing_primary_raises(self):
        with pytest.raises(ValueError, match="Primary model 'primary' not available"):
            generate_predictions.fn({}, _batch_df(), primary="primary")


class TestSaveAndSummary:
    """Tests for save_predictions and log_batch_summary tasks."""

    def _predictions(self):
        return pd.DataFrame(
            {
                "customer_id": ["a", "b", "c", "d"],
                "churn_probability": [0.1, 0.5, 0.8, 0.95],
                "churn_probability_random_forest": [0.2, 0.4, 0.6, 0.9],
                "churn_prediction": [0, 1, 1, 1],
                "risk_group": ["Low", "Medium", "High", "High"],
                "segment": [0, 1, 1, 0],
            }
        )

    def test_writes_historical_and_latest(self, tmp_path):
        historical = save_predictions.fn(self._predictions(), tmp_path, "20260721T120000")

        assert historical == tmp_path / "churn_predictions_20260721T120000.parquet"
        assert historical.exists()
        assert (tmp_path / "latest_predictions.parquet").exists()
        assert len(pd.read_parquet(tmp_path / "latest_predictions.parquet")) == 4

    def test_summary_counts(self):
        summary = log_batch_summary.fn(self._predictions(), "out.parquet")

        assert summary["n_customers"] == 4
        assert summary["n_predicted_churn"] == 3
        assert summary["risk_groups"] == {"Low": 1, "Medium": 1, "High": 2}
        assert summary["avg_churn_probability"] == pytest.approx(0.5875)
        assert summary["output_path"] == "out.parquet"

    def test_summary_reports_per_model_and_segments(self):
        summary = log_batch_summary.fn(self._predictions(), "out.parquet")

        assert summary["avg_churn_probability_per_model"]["random_forest"] == pytest.approx(0.525)
        assert summary["segments"] == {"0": 2, "1": 2}
