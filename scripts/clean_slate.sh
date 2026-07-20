#!/usr/bin/env bash
# Wipe all MLflow + Prefect state for a clean slate: no experiments, runs, or
# models remembered. Source code is left untouched.
#
# Usage: scripts/clean_slate.sh [-y]
#   -y   skip the confirmation prompt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" != "-y" ]]; then
  read -r -p "Delete ALL MLflow experiments/runs/models and reset Prefect history? [y/N] " ans
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

echo "==> Resetting Prefect database"
# `prefect server database reset` can fail on a broken alembic downgrade, so
# just delete the DB files and rebuild a fresh empty schema.
rm -f ~/.prefect/prefect.db ~/.prefect/prefect.db-shm ~/.prefect/prefect.db-wal
rm -rf ~/.prefect/storage/*
uv run prefect server database upgrade -y >/dev/null

echo "==> Clean slate ready. Restart servers when needed:"
echo "    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo "    uv run prefect server start"
