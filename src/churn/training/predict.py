import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def predict(
    model: Any,
    X: np.ndarray,
    return_proba: bool = False,
) -> dict:
    """
    Makes predictions using a trained model.

    Args:
        model: Trained model (classification or clustering).
        X: Features to predict on.
        return_proba: If True, also return probability estimates (classification only).

    Returns:
        Dictionary containing:
            - predictions: Array of predicted labels/clusters.
            - probabilities: Array of probability estimates (if return_proba=True and available).
            - n_samples: Number of samples predicted.
    """
    predictions = model.predict(X)

    result = {
        "predictions": predictions,
        "n_samples": len(predictions),
    }

    if return_proba and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        result["probabilities"] = probabilities
        # For binary classification, also include probability of positive class
        if probabilities.shape[1] == 2:
            result["churn_probability"] = probabilities[:, 1]

    logger.info(f"Generated predictions for {result['n_samples']} samples")

    if "churn_probability" in result:
        avg_churn_prob = np.mean(result["churn_probability"])
        churn_count = np.sum(predictions == 1)
        logger.info(
            f"  Predicted churners: {churn_count} ({churn_count/len(predictions)*100:.1f}%)"
        )
        logger.info(f"  Average churn probability: {avg_churn_prob:.4f}")

    return result
