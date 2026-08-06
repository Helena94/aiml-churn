"""Weekly parent flow: ETL -> training -> batch prediction + model comparison.

Calls the three existing flows in order. Sequential calls mean a failed stage
raises and nothing downstream runs, so no extra orchestration is needed for it.

Run `python scripts/run_weekly.py` to serve the schedule (needs a Prefect server
or Cloud for the cron to fire); import and call `weekly_pipeline()` to run once.
"""
import sys
from pathlib import Path

# Prefect Cloud runs this entrypoint straight out of a fresh clone, where `churn` is
# importable only if the `-e .` pull step landed in the same interpreter that executes
# the flow — it does not. Pointing at src/ directly makes the import independent of that.
# Inert locally, where the editable install already puts churn on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefect import flow  # noqa: E402
from prefect.client.schemas.schedules import CronSchedule  # noqa: E402
from prefect.logging import get_run_logger  # noqa: E402

from run_batch_prediction import batch_prediction  # noqa: E402
from run_etl import run_etl  # noqa: E402
from run_experiments import run_experiments  # noqa: E402


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
