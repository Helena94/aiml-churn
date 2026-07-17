import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def evaluate_classification_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluates a trained classification model on test data.

    Args:
        model: Trained classification model (GridSearchCV or similar).
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Get probability predictions if available (for ROC-AUC)
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="binary", zero_division=0),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)

    # Add confusion matrix values
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])

    # Log metrics
    logger.info("Classification Evaluation Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics


def get_classification_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str] | None = None,
) -> str:
    """
    Generates a detailed classification report.

    Args:
        model: Trained classification model.
        X_test: Test features.
        y_test: Test labels.
        target_names: Optional list of target class names.

    Returns:
        Classification report as a string.
    """
    y_pred = model.predict(X_test)
    target_names = target_names or ["Not Churned", "Churned"]

    report = classification_report(y_test, y_pred, target_names=target_names)
    logger.info(f"Classification Report:\n{report}")

    return report


def evaluate_clustering_model(
    model: Any,
    X: np.ndarray,
) -> dict:
    """
    Evaluates a trained clustering model.

    Args:
        model: Trained clustering model (GridSearchCV with KMeans or similar).
        X: Features to evaluate clustering on.

    Returns:
        Dictionary containing clustering evaluation metrics.
    """
    labels = model.predict(X)

    # Get the best estimator if it's a GridSearchCV
    best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    # Get the actual KMeans model from the pipeline
    if hasattr(best_model, "named_steps") and "model" in best_model.named_steps:
        kmeans = best_model.named_steps["model"]
    else:
        kmeans = best_model

    n_clusters = len(set(labels))

    metrics = {
        "n_clusters": n_clusters,
        "inertia": float(kmeans.inertia_) if hasattr(kmeans, "inertia_") else None,
        "cluster_sizes": {
            f"cluster_{i}": int(np.sum(labels == i)) for i in range(n_clusters)
        },
    }

    # Calculate silhouette score (requires at least 2 clusters)
    if n_clusters >= 2:
        metrics["silhouette_score"] = silhouette_score(X, labels)

    # Log metrics
    logger.info("Clustering Evaluation Results:")
    logger.info(f"  Number of Clusters: {metrics['n_clusters']}")
    if metrics.get("silhouette_score") is not None:
        logger.info(f"  Silhouette Score:   {metrics['silhouette_score']:.4f}")
    if metrics.get("inertia") is not None:
        logger.info(f"  Inertia:            {metrics['inertia']:.4f}")
    logger.info(f"  Cluster Sizes:      {metrics['cluster_sizes']}")

    return metrics


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray | None,
    model_type: str,
) -> dict:
    """
    Unified evaluation function that routes to the appropriate evaluator.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels (optional for clustering).
        model_type: Type of model ('classification' or 'clustering').

    Returns:
        Dictionary containing evaluation metrics.
    """
    if model_type == "clustering":
        return evaluate_clustering_model(model, X_test)
    else:
        if y_test is None:
            raise ValueError("y_test is required for classification evaluation")
        return evaluate_classification_model(model, X_test, y_test)
