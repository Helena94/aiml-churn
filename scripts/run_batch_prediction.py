"""Entry point for the telco churn batch prediction flow."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from prefect import flow
from prefect.logging import get_run_logger

from churn.prediction.batch import (
    add_prediction_metadata,
    enrich_with_source,
    generate_predictions,
    load_batch_data,
    load_mlflow_model,
    log_batch_summary,
    model_comparison,
    prepare_batch_input,
    resolve_model_version,
    save_comparison,
    save_predictions,
    validate_batch_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "batch_prediction.yml"
PRIMARY = "primary"


def _resolve(path: str | Path) -> Path:
    """Resolve a config path against the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@flow(name="telco-churn-batch-prediction")
def batch_prediction(config_path: str | Path = DEFAULT_CONFIG, **overrides: Any) -> dict:
    """
    Score a batch of customers with the registered MLflow models, then save the
    multi-model comparison report alongside the predictions.

    Args:
        config_path: Path to the batch prediction YAML config.
        **overrides: Config keys to override (e.g. input_path, decision_threshold).

    Returns:
        The batch summary dictionary.
    """
    logger = get_run_logger()
    with open(_resolve(config_path)) as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)

    batch_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    logger.info(f"Starting batch prediction {batch_id}")

    input_path = _resolve(cfg["input_path"])
    # Doubles as the enrichment source below: the same cleaned dataset that stands in
    # for an extract is the only place the true labels live.
    fallback_source = _resolve(cfg["fallback_source"]) if cfg.get("fallback_source") else None
    if fallback_source:
        prepare_batch_input(input_path, fallback_source)

    df = load_batch_data(input_path)
    validate_batch_data(df)

    tracking_uri = cfg.get("tracking_uri")
    primary_uri = cfg["primary_model_uri"]

    # The primary model is required; the comparison and segmentation models are
    # extra analytical columns, so a missing one degrades the output instead of
    # failing the whole batch.
    models = {PRIMARY: load_mlflow_model(primary_uri, tracking_uri)}

    optional = dict(cfg.get("comparison_models") or {})
    segmentation = cfg.get("segmentation_model_uri")
    if segmentation:
        optional["segment"] = segmentation

    for name, uri in optional.items():
        try:
            models[name] = load_mlflow_model(uri, tracking_uri)
        except Exception as exc:
            logger.warning(f"Skipping optional model '{name}' ({uri}): {exc}")

    predictions = generate_predictions(
        models,
        df,
        primary=PRIMARY,
        segmentation="segment" if "segment" in models else None,
        decision_threshold=cfg["decision_threshold"],
        risk_low=cfg["risk_low"],
        risk_high=cfg["risk_high"],
    )
    predictions = add_prediction_metadata(
        predictions,
        model_uri=primary_uri,
        batch_id=batch_id,
        model_version=resolve_model_version(primary_uri),
    )
    predictions = enrich_with_source(predictions, fallback_source)

    output_dir = _resolve(cfg["output_dir"])
    output_path = save_predictions(predictions, output_dir, batch_id)

    comparison = model_comparison(
        predictions,
        model_uris=dict(cfg.get("comparison_models") or {}),
        decision_threshold=cfg["decision_threshold"],
        risk_low=cfg["risk_low"],
        risk_high=cfg["risk_high"],
    )
    comparison_path = save_comparison(comparison, output_dir, batch_id)

    summary = log_batch_summary(predictions, output_path)
    summary["comparison_path"] = str(comparison_path)
    return summary


if __name__ == "__main__":
    batch_prediction()
