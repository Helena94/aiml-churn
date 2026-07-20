"""Unit tests for training evaluation functions."""

import numpy as np
import pytest

from churn.training.evaluation import (
    evaluate_classification_model,
    evaluate_clustering_model,
    evaluate_model,
    get_classification_report,
)


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


class StubKMeans:
    """Fake clustering model returning fixed labels plus inertia_."""

    def __init__(self, labels, inertia=12.5):
        self._labels = np.asarray(labels)
        self.inertia_ = inertia

    def predict(self, X):
        return self._labels


class TestEvaluateClassificationModel:
    """Tests for evaluate_classification_model function."""

    def test_confusion_matrix_values(self):
        # y_true vs y_pred chosen for a known confusion matrix:
        # tn=2, fp=1, fn=1, tp=2
        y_test = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])
        X_test = np.zeros((6, 1))
        model = StubClassifier(y_pred)

        result = evaluate_classification_model(model, X_test, y_test)

        assert result["true_negatives"] == 2
        assert result["false_positives"] == 1
        assert result["false_negatives"] == 1
        assert result["true_positives"] == 2
        assert result["accuracy"] == pytest.approx(4 / 6)

    def test_roc_auc_present_with_proba(self):
        y_test = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
        model = StubProbaClassifier(y_pred, proba)

        result = evaluate_classification_model(model, np.zeros((4, 1)), y_test)

        assert "roc_auc" in result

    def test_roc_auc_absent_without_proba(self):
        y_test = np.array([0, 1])
        model = StubClassifier(np.array([0, 1]))

        result = evaluate_classification_model(model, np.zeros((2, 1)), y_test)

        assert "roc_auc" not in result


class TestGetClassificationReport:
    """Tests for get_classification_report function."""

    def test_returns_string_with_default_names(self):
        y_test = np.array([0, 0, 1, 1])
        model = StubClassifier(np.array([0, 0, 1, 1]))

        report = get_classification_report(model, np.zeros((4, 1)), y_test)

        assert isinstance(report, str)
        assert "Not Churned" in report
        assert "Churned" in report


class TestEvaluateClusteringModel:
    """Tests for evaluate_clustering_model function."""

    def test_two_cluster_metrics(self):
        labels = np.array([0, 0, 1, 1])
        X = np.array([[0.0], [0.1], [5.0], [5.1]])
        model = StubKMeans(labels, inertia=3.0)

        result = evaluate_clustering_model(model, X)

        assert result["n_clusters"] == 2
        assert result["cluster_sizes"] == {"cluster_0": 2, "cluster_1": 2}
        assert result["inertia"] == pytest.approx(3.0)
        assert "silhouette_score" in result


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_routes_to_clustering(self):
        labels = np.array([0, 0, 1, 1])
        X = np.array([[0.0], [0.1], [5.0], [5.1]])
        model = StubKMeans(labels)

        result = evaluate_model(model, X, None, "clustering")

        assert result["n_clusters"] == 2

    def test_classification_requires_y_test(self):
        model = StubClassifier(np.array([0, 1]))

        with pytest.raises(ValueError, match="y_test is required"):
            evaluate_model(model, np.zeros((2, 1)), None, "classification")
