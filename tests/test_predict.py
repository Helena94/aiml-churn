"""Unit tests for training predict function."""

import numpy as np

from churn.training.predict import predict


class StubClassifier:
    """Fake classifier returning fixed predictions, no predict_proba."""

    def __init__(self, predictions):
        self._predictions = np.asarray(predictions)

    def predict(self, X):
        return self._predictions


class StubProbaClassifier(StubClassifier):
    """Fake classifier that also exposes predict_proba."""

    def __init__(self, predictions, proba):
        super().__init__(predictions)
        self._proba = np.asarray(proba)

    def predict_proba(self, X):
        return self._proba


class TestPredict:
    """Tests for predict function."""

    def test_n_samples_and_no_proba_by_default(self):
        model = StubClassifier([0, 1, 0])

        result = predict(model, np.zeros((3, 1)))

        assert result["n_samples"] == 3
        assert result["predictions"].tolist() == [0, 1, 0]
        assert "probabilities" not in result

    def test_no_proba_when_model_lacks_predict_proba(self):
        model = StubClassifier([0, 1])

        result = predict(model, np.zeros((2, 1)), return_proba=True)

        assert "probabilities" not in result

    def test_churn_probability_is_positive_class_column(self):
        proba = np.array([[0.8, 0.2], [0.3, 0.7]])
        model = StubProbaClassifier([0, 1], proba)

        result = predict(model, np.zeros((2, 1)), return_proba=True)

        assert result["churn_probability"].tolist() == [0.2, 0.7]
