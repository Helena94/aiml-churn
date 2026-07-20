"""Unit tests for training dispatcher."""

import pytest

from churn.training.train import train_model


class TestTrainModel:
    """Tests for train_model function."""

    def test_unknown_model_raises(self):
        # Builders are never reached, so args can be trivial (no fitting).
        with pytest.raises(ValueError, match="Unsupported model type: unknown_model"):
            train_model(None, None, "unknown_model", {})
