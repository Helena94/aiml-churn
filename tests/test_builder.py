"""Unit tests for feature builder functions."""

import pandas as pd
import pytest

from churn.features.builder import build_features, split_features_target


def _sample_df():
    return pd.DataFrame(
        {
            "customer_id": ["a", "b"],
            "gender": ["male", "female"],
            "monthly_charges": [10.0, 20.0],
            "churn": ["No", "Yes"],
        }
    )


class TestSplitFeaturesTarget:
    """Tests for split_features_target function."""

    def test_drops_id_and_target_from_X(self):
        X, y = split_features_target(_sample_df())

        assert "churn" not in X.columns
        assert "customer_id" not in X.columns
        assert "gender" in X.columns
        assert "monthly_charges" in X.columns
        assert y.name == "churn"
        assert y.tolist() == ["No", "Yes"]

    def test_missing_target_raises(self):
        df = _sample_df().drop(columns=["churn"])

        with pytest.raises(KeyError):
            split_features_target(df)


class TestBuildFeatures:
    """Tests for build_features function."""

    def test_drops_id_and_target(self):
        X = build_features(_sample_df())

        assert "customer_id" not in X.columns
        assert "churn" not in X.columns
        assert "gender" in X.columns

    def test_tolerates_missing_id_and_target(self):
        # Inference frames may not carry customer_id / churn.
        df = _sample_df().drop(columns=["customer_id", "churn"])

        X = build_features(df)

        assert list(X.columns) == ["gender", "monthly_charges"]
