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
    enrich_with_source,
    generate_predictions,
    load_batch_data,
    log_batch_summary,
    model_comparison,
    prepare_batch_input,
    save_comparison,
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
    """Fake clustering model — labels and centroid distances, no predict_proba."""

    def __init__(self, labels, distances=None):
        self._labels = np.asarray(labels)
        # (n_samples, n_clusters), as sklearn's KMeans.transform returns.
        self._distances = (
            np.zeros((len(self._labels), 2)) if distances is None else np.asarray(distances)
        )

    def predict(self, X):
        return self._labels

    def transform(self, X):
        return self._distances


class StubSegmenterNoTransform:
    """Clustering model that cannot report distances (no `transform` at all)."""

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

    def test_segment_distance_is_the_nearest_centroid(self):
        models = self._models()
        models["segment"] = StubSegmenter(
            [0, 1, 1, 0],
            distances=[[0.5, 2.0], [3.0, 0.25], [4.0, 1.5], [0.75, 6.0]],
        )
        out = generate_predictions.fn(
            models, _batch_df(), primary="primary", segmentation="segment"
        )

        assert out["segment_distance"].tolist() == [0.5, 0.25, 1.5, 0.75]

    def test_segment_distance_omitted_without_transform(self):
        models = self._models()
        models["segment"] = StubSegmenterNoTransform([0, 1, 1, 0])
        out = generate_predictions.fn(
            models, _batch_df(), primary="primary", segmentation="segment"
        )

        assert out["segment"].tolist() == [0, 1, 1, 0]
        assert "segment_distance" not in out.columns

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


class TestEnrichWithSource:
    """Tests for enrich_with_source task."""

    def _source(self, tmp_path, ids):
        """Cleaned-CSV stand-in: features plus the identifier and target columns."""
        df = pd.DataFrame({"customer_id": list(ids)})
        for col in FEATURE_COLUMNS:
            df[col] = list(range(len(df)))
        df["churn"] = ["Yes" if i % 2 == 0 else "No" for i in range(len(df))]
        path = tmp_path / "cleaned.csv"
        df.to_csv(path, index=False)
        return path

    def _predictions(self, ids=("id-2", "id-0")):
        return pd.DataFrame(
            {"customer_id": list(ids), "churn_probability": [0.8, 0.1]},
        )

    def test_attaches_label_and_features(self, tmp_path):
        source = self._source(tmp_path, [f"id-{i}" for i in range(4)])

        out = enrich_with_source.fn(self._predictions(), source)

        assert out["churn_actual"].tolist() == ["Yes", "Yes"]
        assert set(FEATURE_COLUMNS).issubset(out.columns)
        assert "churn" not in out.columns  # renamed, never shadows the predicted label

    def test_scored_rows_and_order_are_preserved(self, tmp_path):
        source = self._source(tmp_path, [f"id-{i}" for i in range(4)])

        out = enrich_with_source.fn(self._predictions(), source)

        # Only the two scored customers, in the order they were scored.
        assert out["customer_id"].tolist() == ["id-2", "id-0"]
        assert out["churn_probability"].tolist() == [0.8, 0.1]

    def test_unmatched_customer_gets_a_null_label(self, tmp_path):
        source = self._source(tmp_path, ["id-0", "id-1"])

        out = enrich_with_source.fn(self._predictions(ids=("id-0", "id-99")), source)

        assert len(out) == 2
        assert out["churn_actual"].isna().tolist() == [False, True]

    def test_no_source_returns_predictions_unchanged(self):
        predictions = self._predictions()

        out = enrich_with_source.fn(predictions, None)

        assert out is predictions


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

    def test_writes_csv_alongside_parquet(self, tmp_path):
        save_predictions.fn(self._predictions(), tmp_path, "20260721T120000")

        assert (tmp_path / "churn_predictions_20260721T120000.csv").exists()
        latest = pd.read_csv(tmp_path / "latest_predictions.csv")
        assert len(latest) == 4
        assert latest["churn_probability"].tolist() == [0.1, 0.5, 0.8, 0.95]

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


class TestModelComparison:
    """Tests for the model_comparison and save_comparison tasks."""

    def _scored(self):
        """Three classifiers over four customers, with hand-checkable votes.

        At threshold 0.5 the rows are: all-agree-no-churn, all-agree-churn, then
        two splits. random_forest is the deliberate outlier on both splits.
        """
        return pd.DataFrame(
            {
                "customer_id": ["a", "b", "c", "d"],
                "churn_probability_logistic_regression": [0.10, 0.60, 0.10, 0.90],
                "churn_probability_random_forest": [0.20, 0.70, 0.60, 0.40],
                "churn_probability_xgboost": [0.30, 0.80, 0.20, 0.95],
                # The primary's duplicate column must not be counted as a model.
                "churn_probability": [0.30, 0.80, 0.20, 0.95],
                "segment": [0, 1, 1, 0],
            }
        )

    def _row(self, comparison, model):
        return comparison.set_index("model").loc[model]

    def test_one_row_per_classifier(self):
        comparison = model_comparison.fn(self._scored())

        # `churn_probability` lacks the trailing underscore, so it is excluded.
        assert list(comparison["model"]) == [
            "logistic_regression",
            "random_forest",
            "xgboost",
        ]

    def test_per_model_distribution(self):
        row = self._row(model_comparison.fn(self._scored()), "logistic_regression")

        assert row["mean_churn_probability"] == pytest.approx(0.425)
        assert row["median_churn_probability"] == pytest.approx(0.35)
        assert row["n_predicted_churn"] == 2
        assert row["churn_rate"] == pytest.approx(0.5)

    def test_per_model_risk_split(self):
        comparison = model_comparison.fn(self._scored())
        rf = self._row(comparison, "random_forest")

        # 0.20 -> Low, 0.70 -> High, 0.60 -> Medium, 0.40 -> Medium
        assert (rf["risk_low"], rf["risk_medium"], rf["risk_high"]) == (1, 2, 1)

    def test_agreement_with_consensus_flags_the_outlier(self):
        comparison = model_comparison.fn(self._scored())

        # Majority vote is no/yes/no/yes; rf disagrees on both split rows.
        assert self._row(comparison, "logistic_regression")["agreement_with_consensus"] == 1.0
        assert self._row(comparison, "xgboost")["agreement_with_consensus"] == 1.0
        assert self._row(comparison, "random_forest")["agreement_with_consensus"] == pytest.approx(
            0.5
        )

    def test_batch_agreement_stats_repeat_on_every_row(self):
        comparison = model_comparison.fn(self._scored())

        assert list(comparison["batch_all_agree_churn"]) == [1, 1, 1]
        assert list(comparison["batch_all_agree_no_churn"]) == [1, 1, 1]
        assert list(comparison["batch_disagree"]) == [2, 2, 2]
        assert comparison["batch_agreement_rate"].iloc[0] == pytest.approx(0.5)
        assert comparison["batch_n_customers"].iloc[0] == 4
        # Spreads 0.20, 0.20, 0.50, 0.55
        assert comparison["batch_mean_probability_spread"].iloc[0] == pytest.approx(0.3625)

    def test_decision_threshold_is_honoured(self):
        comparison = model_comparison.fn(self._scored(), decision_threshold=0.95)

        assert self._row(comparison, "xgboost")["n_predicted_churn"] == 1
        assert self._row(comparison, "logistic_regression")["n_predicted_churn"] == 0

    def test_lineage_columns(self):
        comparison = model_comparison.fn(
            self._scored(), model_uris={"random_forest": "runs:/abc/model"}
        )
        rf = self._row(comparison, "random_forest")

        assert rf["model_uri"] == "runs:/abc/model"
        # Not a `models:/name@alias` URI, so no registry lookup is attempted.
        assert rf["model_version"] is None
        assert self._row(comparison, "xgboost")["model_uri"] is None

    def test_no_classifier_columns_raises(self):
        with pytest.raises(ValueError, match="No 'churn_probability_"):
            model_comparison.fn(pd.DataFrame({"customer_id": ["a"], "segment": [0]}))

    def test_save_writes_historical_and_latest(self, tmp_path):
        comparison = model_comparison.fn(self._scored())
        historical = save_comparison.fn(comparison, tmp_path, "20260721T120000")

        assert historical == tmp_path / "model_comparison_20260721T120000.csv"
        assert (tmp_path / "latest_model_comparison.csv").exists()

        saved = pd.read_csv(tmp_path / "latest_model_comparison.csv")
        assert len(saved) == 3
        assert list(saved["batch_id"]) == ["20260721T120000"] * 3
        assert "comparison_timestamp" in saved.columns
