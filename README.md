# Telco Customer Churn Prediction

End-to-end ML pipeline for predicting customer churn on the Kaggle
[`blastchar/telco-customer-churn`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset. Covers ETL (Prefect-orchestrated), feature engineering, model training with
hyperparameter search, experiment tracking + model registry (MLflow), and batch scoring.

**Setup guides:** [RUN_LOCAL.md](RUN_LOCAL.md) — everything on your machine.
[RUN_PREFECT_CLOUD.md](RUN_PREFECT_CLOUD.md) — same code orchestrated by Prefect Cloud, either
executing locally or in a managed container. [SWITCHING.md](SWITCHING.md) — the two independent
switches (Prefect profile and `MLFLOW_TRACKING_URI`) and how to tell where you are pointed.

## Run order

The stages are strictly sequential — each one consumes the previous one's output.

| # | Step | Command | Produces |
|---|------|---------|----------|
| 0 | Install deps | `uv sync --group dev` | `.venv/` |
| 0 | Kaggle credentials | put `kaggle.json` in `kaggle/`, or set `KAGGLE_USERNAME` + `KAGGLE_KEY` | — |
| 1 | ETL | `python scripts/run_etl.py` | `data/processed/churn_cleaned.csv` |
| 2 | Train + register | `python scripts/run_experiments.py` | MLflow runs + registered models in `mlflow.db` |
| 3 | Batch scoring + comparison | `python scripts/run_batch_prediction.py` | `data/predictions/*.parquet` + `*.csv` + `model_comparison_*.csv` |

Skipping a step fails the next one: without step 1 training has no data, without step 2
the `models:/...@champion` URIs don't resolve.

### Weekly schedule

`scripts/run_weekly.py` is a parent flow that runs steps 1–3 in order as nested subflows.
The calls are sequential, so a failed stage stops everything downstream.

```bash
uv run prefect profile use local   # CLI and serve() must share a backend — see SWITCHING.md
uv run prefect server start        # separate terminal — needed for the schedule to fire
uv run python scripts/run_weekly.py   # keep running: serve() IS the deployment
```

That serves a `weekly-churn-analysis` deployment on cron `0 9 * * 1`
(Mondays 09:00 `Europe/Belgrade`) with `limit=1`, so runs can't overlap. No work pool or
worker is needed — `serve()` runs its own runner. Trigger it immediately instead of waiting
for Monday:

```bash
uv run prefect deployment run 'weekly-churn-analysis/weekly-churn-analysis'
```

Because the Kaggle dataset is static, the schedule demonstrates Prefect orchestration — it
is not evidence that weekly retraining improves the models.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                            # runtime deps — what the flows import
uv sync --group dev                # + pytest, jupyter, ipykernel
uv sync --group dev --group eda    # + the notebook-only EDA stack
```

The `eda` group (`ydata-profiling`, `sweetviz`, `squarify`, `anthropic`) is imported only by
the notebooks, never by `src/` or `scripts/`. Keeping it out of the runtime dependencies is
what lets the pipeline install inside Prefect Cloud's 2 GB managed container — see
[RUN_PREFECT_CLOUD.md](RUN_PREFECT_CLOUD.md).

Kaggle credentials are required for the ETL download step. Put them in `kaggle/kaggle.json`,
or set `KAGGLE_USERNAME` and `KAGGLE_KEY` directly — `etl/load_kaggle_credentials.py` prefers
the environment and falls back to the file, so a deployment can supply them as secrets.

Environment variables the pipeline reads, all optional:

| Variable | Effect when unset |
|----------|-------------------|
| `MLFLOW_TRACKING_URI` | Falls back to the project-local `mlflow.db` |
| `MLFLOW_TRACKING_USERNAME` / `_PASSWORD` | Only needed by an authenticated remote tracking server |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Falls back to `kaggle/kaggle.json` |

`.env` is optional and only holds `ANTHROPIC_API_KEY`; no pipeline stage needs it.

`MLFLOW_TRACKING_URI` is one of the two switches between a local and a cloud setup — it is
per-shell and silent, so a leftover `export` sends "local" runs to a remote server with no
visible sign. [SWITCHING.md](SWITCHING.md) covers it and the Prefect profile together.

## Pipeline

### 1. ETL (Prefect)

Extract → transform → load. Downloads the raw dataset, cleans/validates it, and writes
`data/processed/churn_cleaned.csv` plus `transformation_log.json`.

```bash
python scripts/run_etl.py
```

The transform step caches for 3 days keyed on the raw data hash — re-running is cheap
and won't redo the work unless the source data changed.

### 2. Train & register (Prefect + MLflow)

Trains all four models (logistic regression, random forest, XGBoost, kmeans) in parallel,
logging params, metrics, and the fitted pipeline to MLflow.

```bash
python scripts/run_experiments.py
```

Each model's hyperparameter grid lives in `configs/<model>.yml`.

Preprocessing lives **inside** each model pipeline, so the logged artifact scores raw
frames — no train/serve skew.

After training, every model is registered under its own name with a `champion` alias:

| Model | Registered name |
|-------|-----------------|
| logistic regression | `telco-churn-logistic-regression` |
| random forest | `telco-churn-random-forest` |
| XGBoost | `telco-churn-xgboost` |
| kmeans | `telco-churn-segmentation` |

The best classifier by `roc_auc` is additionally registered as `telco-churn-model` and
aliased `@champion` — that's the primary model the batch flow scores with. kmeans never
competes for primary (it has no `roc_auc`).

### Batch input

The batch flow reads `data/processed/batch_customers.parquet` (path set in
`configs/batch_prediction.yml`; `.csv` is also accepted). It stands in for whatever customer
extract you want scored.

When that file is missing, the flow's first task builds it from `fallback_source` and writes
**only the held-out test split** — the same 20% `run_experiments.py` kept out of training, so
the batch scores customers the models have never seen. An existing file is rebuilt only once
`fallback_source` is newer than it: a fresh ETL run re-splits the holdout, so keeping last
week's batch would score rows the models were just trained on. Point a real pipeline at
`input_path` and set `fallback_source: null` to turn the stand-in off — the task is then
skipped entirely and your file is never touched; a missing input fails the run.

The file must carry `customer_id` plus every column in `CATEGORICAL_COLUMNS` and
`NUMERICAL_COLUMNS` from `features/schema.py`, with no duplicate IDs — `validate_batch_data`
rejects it otherwise. The `churn` column is not required and is ignored if present — the true label reaches the
output by being merged back from `fallback_source` after scoring, not by riding in on the
input.

### 3. Batch prediction (Prefect + MLflow)

Loads the registered models and scores the batch. Never retrains.

```bash
python scripts/run_batch_prediction.py
```

Writes `data/predictions/churn_predictions_<batch_id>.parquet` and
`latest_predictions.parquet`, plus a `.csv` twin of each — same content, one for code to
read and one to open or plot from.

Prediction columns: `churn_probability_<name>` per classifier, `segment` and
`segment_distance` from kmeans, and from the primary model `churn_probability`,
`churn_prediction`, `churn_label`, `risk_group` (Low <0.40, Medium 0.40–0.69, High ≥0.70),
plus lineage columns (`prediction_timestamp`, `model_uri`, `model_version`, `batch_id`).

`segment_distance` is the distance to the customer's assigned centroid in the model's
preprocessed space — how typical they are of their segment. Omitted if the segmentation
model has no `transform`.

Each row also carries its full source row merged back from `fallback_source` on
`customer_id`: every feature column plus the true label as **`churn_actual`** (renamed so it
can't be confused with `churn_label`, which is *predicted*). That is what makes the output
self-contained — error metrics and feature-sliced plots need no second file:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

df = pd.read_csv("data/predictions/latest_predictions.csv")
y = (df["churn_actual"] == "Yes").astype(int)
roc_auc_score(y, df["churn_probability"])          # 0.8467 — matches MLflow's champion
confusion_matrix(y, df["churn_prediction"])
pd.crosstab(df["segment"], df["churn_actual"])     # do the clusters split on churn?
```

Because the batch is the same held-out 20% `run_experiments.py` kept out of training, metrics
computed here reproduce the champion's test metrics in MLflow exactly.

Enrichment reuses the `fallback_source` key — no separate setting. Set it to `null` (the real
extract case, where no labels exist) and the merge is skipped: the output keeps only
`customer_id` and the prediction columns. `src/churn/training/evaluation.py` already has
`evaluate_classification_model()` if you'd rather not hand-roll the metrics.

A missing optional model is logged and skipped; a missing primary is a hard error.
Thresholds, model URIs, and paths all live in `configs/batch_prediction.yml`.

### 4. Model comparison report

Runs automatically at the end of the batch flow — no separate command. Derived entirely from
the prediction columns, so no model is reloaded and nothing is re-scored.

Writes `data/predictions/model_comparison_<batch_id>.csv` and
`latest_model_comparison.csv`: one row per classifier with its probability distribution
(`mean_churn_probability`, `median_churn_probability`), `n_predicted_churn`, `churn_rate`,
risk split (`risk_low` / `risk_medium` / `risk_high`), registry lineage (`model_uri`,
`model_version`), and `agreement_with_consensus` — how often that model matches the majority
vote, which is what flags an outlier model.

Batch-level agreement statistics are repeated on every row as `batch_*` columns
(`batch_all_agree_churn`, `batch_all_agree_no_churn`, `batch_disagree`,
`batch_agreement_rate`, `batch_mean_probability_spread`). Constant across a three-row table,
which keeps the whole report in one file.

The report needs at least one `churn_probability_<name>` column and raises otherwise. The
primary model's own `churn_probability` column is excluded — it's a duplicate of whichever
classifier is champion.

## UIs

### MLflow

Runs and the model registry live in a local SQLite store (`mlflow.db`).

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Prefect

Optional — the scripts run fine without a server (ephemeral mode). Start one only if you
want the run history UI:

```bash
uv run prefect server start
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

The work pool and worker are only needed if you deploy the flows rather than running the
scripts directly:

```bash
uv run prefect work-pool create local-process --type process --set-as-default
uv run prefect worker start --pool local-process
```

## Reset

`scripts/clean_slate.sh` returns the project to a pre-training state — useful when the
registry is in a confusing state (stale aliases, half-registered models, experiments from
an older schema) or when you want a reproducible run from scratch.

```bash
scripts/clean_slate.sh       # prompts for confirmation
scripts/clean_slate.sh -y    # skip the prompt
```

What it does, in order:

| Step | Action |
|------|--------|
| 1 | Kills any running `prefect server`, `prefect worker`, `mlflow`, `huey_consumer` process (failures ignored — nothing running is fine) |
| 2 | Deletes the MLflow tracking DBs and artifact stores: `mlflow.db`, `src/mlflow.db`, `mlruns/`, `scripts/mlruns/`, `mlartifacts/` |
| 3 | Deletes batch outputs: `data/predictions/churn_predictions_*.{parquet,csv}`, `latest_predictions.{parquet,csv}`, `model_comparison_*.csv`, `latest_model_comparison.csv` |
| 4 | Deletes `~/.prefect/prefect.db*` and `~/.prefect/storage/*`, then rebuilds an empty schema with `prefect server database upgrade` |

**Kept:** all source code, `data/raw/`, `data/processed/` (including `churn_cleaned.csv`
and your `batch_customers.parquet`), `configs/`, `notebooks/`, `reports/`.

**Destroyed:** every experiment, run, metric, logged artifact, and registered model —
including all `@champion` aliases — plus every prediction file and all Prefect run history.

Step 3 is not optional cleanup: prediction and comparison files carry `model_uri` and
`model_version` lineage columns pointing at registry versions that step 2 just deleted, so
leaving them behind means a `latest_predictions.parquet` that silently misattributes its
scores.

Because the ETL output survives, the recovery path skips step 1:

```bash
python scripts/run_experiments.py         # rebuild the registry
python scripts/run_batch_prediction.py    # re-score
```

The Prefect DB is deleted rather than reset via `prefect server database reset`, which can
fail on a broken alembic downgrade.

## Tests

Covers ETL (extract, transform, load), features, training, and batch prediction. Prefect-
decorated functions are tested via `.fn()` to bypass the Prefect runtime.

```bash
pytest                           # all tests
pytest tests/test_transform.py   # one file
```

## Layout

```
src/churn/
├── assets.py     # Prefect asset keys + metadata helper (the lineage graph)
├── etl/          # Prefect tasks/flows: download, extract, transform, load
├── features/     # schema.py (column defs), preprocessing.py (ColumnTransformer), builder.py (split/build)
├── models/       # one builder per model → Pipeline(preprocessing → model) wrapped in a CV search
├── training/     # train.py (dispatch), evaluation.py (metrics), predict.py
├── prediction/   # batch.py — batch scoring tasks
└── flows/        # mlflow.py — train, log and register a single model run

scripts/          # run_etl.py, run_experiments.py, run_batch_prediction.py, run_weekly.py, clean_slate.sh
configs/          # per-model hyperparameter grids + batch_prediction.yml
data/             # raw/ → processed/ → predictions/
```

The column schema is defined once in `features/schema.py`; add/remove columns there. Asset keys
are defined once in `assets.py` for the same reason.

## Assets (Prefect Cloud)

Every output of the pipeline is declared as a Prefect **asset**, so the three flows appear as one
connected lineage graph in the Cloud UI — from `kaggle://blastchar/telco-customer-churn` through
the cleaned CSV and the five registered models to `latest_model_comparison.csv` — each carrying
metadata such as row counts, model versions and `roc_auc`. Assets require a Cloud backend;
locally the decorators behave as ordinary tasks. Details in
[RUN_PREFECT_CLOUD.md](RUN_PREFECT_CLOUD.md#5-assets--the-lineage-graph).

## Scope and future work

This is a **scheduled offline analytical ML workflow**, not a production prediction service.
It ends at batch prediction and multi-model comparison — there is no API, no online serving,
and no automated retraining trigger.

Deliberately left as future work:

| Item | Status |
|------|--------|
| Monitoring (EvidentlyAI) | `src/churn/monitor.py` is an empty placeholder. Would run after prediction, comparing each batch against the cleaned training data as reference. |
| Dashboard (Streamlit) | Would read the saved outputs (`latest_predictions.parquet`, `latest_model_comparison.csv`, MLflow metrics) read-only, without triggering flows. |
| Remaining items of `WEEKLY_PIPELINE_PLAN.md` | Items 4–9 — see the status header in that file for what was built and why the rest was skipped. |
