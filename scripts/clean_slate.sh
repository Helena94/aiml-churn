#!/usr/bin/env bash
# Wipe all MLflow + Prefect state and batch prediction outputs for a clean
# slate: no experiments, runs, models, or predictions remembered. Source code
# and pipeline inputs (data/raw, data/processed) are left untouched.
#
# Usage: scripts/clean_slate.sh [-y]
#   -y   skip the confirmation prompt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" != "-y" ]]; then
  read -r -p "Delete ALL MLflow experiments/runs/models, batch predictions, and reset Prefect history? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 1; }
fi

echo "==> Stopping any running MLflow/Prefect servers"
# Ignore failures: nothing running is fine.
pkill -f "prefect server"  || true
pkill -f "prefect worker"  || true
pkill -f "mlflow"          || true
pkill -f "huey_consumer"   || true
sleep 2

echo "==> Deleting MLflow tracking DBs and artifact stores"
rm -rf ./mlflow.db ./src/mlflow.db ./mlruns ./scripts/mlruns ./mlartifacts

echo "==> Deleting batch prediction outputs"
# Derived from the registry we just wiped — every model_uri/model_version stamp
# in them now points at a version that no longer exists. Inputs are kept.
rm -f ./data/predictions/churn_predictions_*.parquet ./data/predictions/latest_predictions.parquet

echo "==> Resetting Prefect database"
# `prefect server database reset` can fail on a broken alembic downgrade, so
# just delete the DB files and rebuild a fresh empty schema.
rm -f ~/.prefect/prefect.db ~/.prefect/prefect.db-shm ~/.prefect/prefect.db-wal
rm -rf ~/.prefect/storage/*
uv run prefect server database upgrade -y >/dev/null

echo "==> Clean slate ready. Restart servers when needed:"
echo "    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo "    uv run prefect server start"
echo "==> Re-run training before scoring again — the registry is empty:"
echo "    python scripts/run_experiments.py"
echo "    python scripts/run_batch_prediction.py"
