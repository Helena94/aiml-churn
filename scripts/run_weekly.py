"""Weekly parent flow: ETL -> training -> batch prediction + model comparison.

Calls the three existing flows in order. Sequential calls mean a failed stage
raises and nothing downstream runs, so no extra orchestration is needed for it.

Run `python scripts/run_weekly.py` to serve the schedule (needs a Prefect server
or Cloud for the cron to fire); import and call `weekly_pipeline()` to run once.
"""
from prefect import flow
from prefect.client.schemas.schedules import CronSchedule
from prefect.logging import get_run_logger

from run_batch_prediction import batch_prediction
from run_etl import run_etl
from run_experiments import run_experiments


@flow(name="weekly-churn-analysis", log_prints=True)
def weekly_pipeline() -> dict:
    """Run the full offline workflow end to end. Returns the batch summary."""
    logger = get_run_logger()

    logger.info("Stage 1/3: ETL")
    run_etl()

    logger.info("Stage 2/3: training + registry")
    run_experiments()

    logger.info("Stage 3/3: batch prediction + model comparison")
    summary = batch_prediction()

    logger.info(f"Weekly pipeline complete: {summary}")
    return summary


if __name__ == "__main__":
    weekly_pipeline.serve(
        name="weekly-churn-analysis",
        schedule=CronSchedule(cron="0 9 * * 1", timezone="Europe/Belgrade"),
        limit=1,  # no overlapping runs
    )
