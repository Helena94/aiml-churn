# Project Context: AI/ML Customer Churn Prediction

## 1. Project Overview

This project focuses on building an end-to-end Machine Learning pipeline for customer churn prediction. It utilizes a
Kaggle dataset and implements standard MLOps practices for data processing, model training, batch inference, monitoring,
and visualization.

## 2. Current Project Status

* **EDA & Raw Data:** ✅ Completed. `notebooks/TelkoChurnEDA.ipynb`, reports in `reports/`.
* **ETL Pipeline (Prefect):** ✅ Completed. Extracted, cleaned, validated, loaded to `data/processed/churn_cleaned.csv`.
* **ML Training (MLflow + Prefect):** ✅ Completed. Four models trained in parallel, metrics/params/artifacts logged to
  MLflow, each registered with a `champion` alias, best classifier promoted to `telco-churn-model@champion`.
* **Batch Predictions (Prefect):** ✅ Completed. `scripts/run_batch_prediction.py` scores a batch with the registered
  models. ⚠️ The input file `data/processed/batch_customers.parquet` is not produced by any script — supply it
  (see §4 step 3).
* **Model Comparison Report:** ✅ Completed. Runs at the end of the batch flow. Writes
  `data/predictions/model_comparison_<batch_id>.csv` + `latest_model_comparison.csv` — one row per classifier
  (probability distribution, churn count, risk split, registry lineage, agreement with the majority vote) plus
  batch-level agreement statistics. Derived from the prediction columns; no model is reloaded.
* **Weekly Orchestration (Prefect):** ✅ Completed. `scripts/run_weekly.py` runs ETL → training → batch prediction as
  nested subflows on cron `0 9 * * 1` (`Europe/Belgrade`), `limit=1`. The static dataset means the schedule
  demonstrates orchestration, not that weekly retraining improves the models.
* **Monitoring (EvidentlyAI):** ⏳ Pending — deliberate future work. `src/churn/monitor.py` exists but is empty.
* **Dashboard (Streamlit):** ⏳ Pending — deliberate future work. The batch flow already writes
  `data/predictions/latest_predictions.parquet` and `latest_model_comparison.csv` as its fixed-name feeds.

**Project boundary.** The project is deliberately closed here as a *scheduled offline analytical ML workflow* — no API,
no online serving, no automated retraining trigger. See `WEEKLY_PIPELINE_PLAN.md` for which proposed items were built
and why the rest were skipped.

## 3. Directory Structure

The repository strictly follows this organizational structure to maintain modularity:

```text
aiml-churn/
├── configs/                # Per-model hyperparameter grids + batch_prediction.yml
├── data/
│   ├── raw/                # Original Kaggle data (read-only)
│   ├── processed/          # ETL output (churn_cleaned.csv) + batch input
│   └── predictions/        # Inference outputs
├── notebooks/              # EDA
├── reports/                # Generated profiling reports
├── scripts/                # Entry points: run_etl, run_experiments, run_batch_prediction, clean_slate.sh
├── src/
│   └── churn/
│       ├── etl/            # ✅ extract.py, transform.py, load.py, download.py
│       ├── features/       # ✅ schema.py (column defs), preprocessing.py, builder.py
│       ├── models/         # ✅ Model definitions only: logistic_regression, random_forest, xgboost, kmeans
│       ├── training/       # ✅ train.py (dispatch), evaluation.py, predict.py
│       ├── prediction/     # ✅ batch.py — batch scoring tasks
│       ├── flows/          # ✅ mlflow.py — train, log and register one run
│       └── monitor.py      # ⏳ empty
└── tests/                  # pytest — ETL, features, training, batch
```

## 4. Execution Order

Strictly sequential; each stage consumes the previous one's output.

```bash
uv sync --group dev                       # 0. deps; Kaggle creds go in kaggle/kaggle.json
python scripts/run_etl.py                 # 1. → data/processed/churn_cleaned.csv
python scripts/run_experiments.py         # 2. → MLflow runs + registered models (mlflow.db)
                                          # 3. → data/processed/batch_customers.parquet (no script — see below)
python scripts/run_batch_prediction.py    # 4. → data/predictions/*.parquet
```

**Step 3** stands in for an external customer extract. To score the cleaned dataset itself:

```bash
python -c "import pandas as pd; \
pd.read_csv('data/processed/churn_cleaned.csv').drop(columns=['churn']) \
  .to_parquet('data/processed/batch_customers.parquet', index=False)"
```

It must carry `customer_id` plus every column in `CATEGORICAL_COLUMNS` and `NUMERICAL_COLUMNS`
(`src/churn/features/schema.py`), with no duplicate IDs.

Failure modes: no step 1 → training has no data; no step 2 → the `models:/...@champion` URIs don't resolve;
no step 3 → the batch flow raises `FileNotFoundError`.

UIs (optional, any time after the relevant stage):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
uv run prefect server start                       # scripts run ephemerally without it
```

## 5. Reset

`scripts/clean_slate.sh` (add `-y` to skip the prompt) returns the project to a pre-training state:
kills running mlflow/prefect processes, then deletes MLflow state (`mlflow.db`, `mlruns/`, `mlartifacts/` —
every run, metric, artifact, registered model and `@champion` alias), batch outputs
(`data/predictions/*.parquet`), and Prefect state (`~/.prefect/prefect.db*`, `~/.prefect/storage/*`,
rebuilt as an empty schema).

Predictions go with the registry deliberately — their `model_uri`/`model_version` lineage columns would
otherwise reference deleted versions.

Kept: source, `configs/`, `data/raw/`, `data/processed/`. Recovery is steps 2 and 4 of §4 — ETL does not
need to re-run.

## 6. Model Registry

| Model | Registered name | Role |
|-------|-----------------|------|
| logistic regression | `telco-churn-logistic-regression` | comparison probability column |
| random forest | `telco-churn-random-forest` | comparison probability column |
| XGBoost | `telco-churn-xgboost` | comparison probability column |
| kmeans | `telco-churn-segmentation` | `segment` column |
| best classifier by `roc_auc` | `telco-churn-model` | primary — owns `churn_probability`, `churn_prediction`, `risk_group` |

All carry a `champion` alias. kmeans never competes for primary (no `roc_auc`). In batch scoring a missing
optional model is logged and skipped; a missing primary is a hard error.
