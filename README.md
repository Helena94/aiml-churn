# Telco Customer Churn Prediction

End-to-end ML pipeline for predicting customer churn on the Kaggle
[`blastchar/telco-customer-churn`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset. Covers ETL (Prefect-orchestrated), feature engineering, model training with
hyperparameter search, and experiment tracking (MLflow).

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # install runtime deps
uv sync --group dev    # + pytest, jupyter, ipykernel
```

Kaggle credentials are required for the ETL download step. Put them in
`kaggle/kaggle.json` (loaded into env vars at runtime).

## Pipeline

### 1. ETL (Prefect)

Extract → transform → load. Downloads the raw dataset, cleans/validates it, and writes
`data/processed/churn_cleaned.csv`.

```bash
python scripts/run_etl.py
```

### 2. Train & track experiments (Prefect + MLflow)

Trains all four models (logistic regression, random forest, XGBoost, kmeans) in parallel,
logging params, metrics, and the fitted model to MLflow.

```bash
python scripts/run_experiments.py
```

Each model's hyperparameter grid lives in `configs/<model>.yml`.

### 3. Inspect runs (MLflow UI)

Runs are tracked in a local SQLite store (`mlflow.db`).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Tests

Covers the ETL functions (extract, transform, load). Prefect-decorated functions are
tested via `.fn()` to bypass the Prefect runtime.

```bash
pytest                           # all tests
pytest tests/test_transform.py   # one file
```

## Layout

```
src/churn/
├── etl/          # Prefect tasks/flows: download, extract, transform, load
├── features/     # schema.py (column defs), preprocessing.py (ColumnTransformer), builder.py (split/encode)
├── models/       # one builder per model → Pipeline(preprocessing → model) wrapped in a CV search
├── training/     # train.py (dispatch), evaluation.py (metrics), predict.py
└── flows/        # mlflow.py — train + log a single model run

scripts/          # run_etl.py, run_experiments.py
configs/          # per-model hyperparameter grids
```

The column schema is defined once in `features/schema.py`; add/remove columns there.
