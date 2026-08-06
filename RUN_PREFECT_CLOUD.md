# Running against Prefect Cloud

Same code, same commands — only the orchestration backend changes. Prefect Cloud gives a
hosted UI, run history, and the schedule; the machine that actually executes the flow depends
on which of the two modes below you use.

For the fully local setup see [RUN_LOCAL.md](RUN_LOCAL.md).

## 0. Prerequisites

Everything from `RUN_LOCAL.md` section 0 (uv, `uv sync --group dev`, `kaggle/kaggle.json`),
plus a free Prefect Cloud account with a workspace.

## 1. Log in and switch backend

```bash
uv run prefect cloud login          # opens a browser, stores an API key in the profile
uv run prefect profile ls           # the * marks the active profile
uv run prefect config view | grep API_URL
```

You are on Cloud when the URL looks like:

```
PREFECT_API_URL='https://api.prefect.cloud/api/accounts/<account-id>/workspaces/<workspace-id>'
```

If a `prefect-cloud` profile already exists, `uv run prefect profile use prefect-cloud` is
enough — no re-login needed. The API key lives in `~/.prefect/profiles.toml`; do not commit it.

## 2. Mode A — execute locally, orchestrate in Cloud (recommended)

The flow runs on this machine, so it has the Kaggle token, the data, and the local `mlflow.db`.
Cloud only schedules it and records the run.

```bash
uv run prefect profile use prefect-cloud
uv run python scripts/run_weekly.py        # keep this terminal running
```

`serve()` registers `weekly-churn-analysis/weekly-churn-analysis` in the Cloud workspace on
cron `0 9 * * 1` (Europe/Belgrade) and runs its own runner — **no work pool, no worker**.
The deployment exists only while this process is alive.

Trigger it now instead of waiting for Monday, from a second terminal:

```bash
uv run prefect deployment ls
uv run prefect deployment run 'weekly-churn-analysis/weekly-churn-analysis'
```

Watch it in the Cloud UI (Runs → the new flow run). Logs also stream in the serving terminal.
MLflow results still land locally — `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

The individual stages work the same way; with the Cloud profile active, any of

```bash
python scripts/run_etl.py
python scripts/run_experiments.py
python scripts/run_batch_prediction.py
```

executes locally and reports to the Cloud UI.

## 3. Mode B — execute on Cloud's managed infrastructure

Nothing of yours runs: Prefect clones the repo into a container it owns, installs the
dependencies, and executes the flow. Your laptop can be closed.

| Field | Value |
|-------|-------|
| Entrypoint | `scripts/run_weekly.py:weekly_pipeline` |
| Work pool | type `prefect:managed` — Prefect runs it, no worker of yours |
| Source | clones the GitHub repo, then `pip install -r requirements.txt` (which is just `-e .`) |
| Schedule | set separately with `prefect-cloud schedule` |

The container is a **fresh clone with a disposable filesystem**, which dictates everything
below: `kaggle/`, `data/` and `*.db` are gitignored, so credentials must arrive as secrets and
anything worth keeping must be written somewhere remote. All three stages run inside one flow
run, so they do share a filesystem for the duration — only the model registry and the outputs
need to outlive it.

### 3.1 One-time setup

**a. A remote MLflow server.** [DagsHub](https://dagshub.com) is free, gives a hosted MLflow
tracking server *and* artifact storage in one signup, and connects to an existing GitHub repo.
Sign up, connect this repo, then take the tracking URL and a token from the repo's *Remote →
Experiments* tab.

Any MLflow server works — `MLFLOW_TRACKING_URI` is all the code reads
(`churn/flows/mlflow.py`). Unset, it still falls back to the local `mlflow.db`, so nothing about
local runs changes.

**b. Verify it locally before spending any Cloud compute.** This is the step that catches
problems for free — the model URIs in `configs/batch_prediction.yml` use aliases
(`models:/<name>@champion`), and an MLflow server that doesn't support aliases would fail
stage 3 only after a full training run:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
python scripts/run_experiments.py && python scripts/run_batch_prediction.py
```

Green, with the models visible in the DagsHub UI ⇒ continue. If aliases turn out to be
unsupported, change those URIs to `models:/<name>/latest` and re-run.

**c. Prefect Cloud + GitHub:**

```bash
uvx prefect-cloud login
uvx prefect-cloud github setup      # only needed for a private repo
```

### 3.2 Deploy

Push to GitHub first — the container clones the repo, so uncommitted work is invisible to it.
Secrets are typed straight into this command; they are stored in the workspace, never in the repo.

```bash
uvx prefect-cloud deploy scripts/run_weekly.py:weekly_pipeline \
  --from <github-user>/<repo> \
  --name weekly-churn-analysis \
  --with-requirements requirements.txt \
  --env MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow \
  --env MLFLOW_TRACKING_USERNAME=<user> \
  --secret MLFLOW_TRACKING_PASSWORD=<dagshub-token> \
  --secret KAGGLE_USERNAME=<kaggle-username> \
  --secret KAGGLE_KEY=<kaggle-key>
```

`--env` values are visible in the UI; `--secret` values are write-only. `KAGGLE_USERNAME` /
`KAGGLE_KEY` are the same two values found in `kaggle/kaggle.json`.

### 3.3 Run it, then schedule it

```bash
uvx prefect-cloud run weekly-churn-analysis/weekly-churn-analysis
uvx prefect-cloud schedule weekly-churn-analysis/weekly-churn-analysis "0 7 * * 1"
```

Verify the manual run first — a broken schedule burns the free compute quota every week and
fills the run history with failures.

`prefect-cloud schedule` takes **no timezone flag**, so the cron is UTC. `0 7 * * 1` is Monday
09:00 in Belgrade during summer (CEST) and 08:00 in winter (CET); the `serve()` schedule in
Mode A is timezone-aware and does not drift. Use the Cloud UI's schedule editor if you need a
timezone attached.

A successful run leaves: three green stages in Prefect Cloud, a new version of each model in
DagsHub, the batch outputs as MLflow artifacts under a `batch-<batch_id>` run, and a connected
asset graph (see §5).

### 3.4 Free-tier limits worth knowing

| Limit | Value | Relevance |
|-------|-------|-----------|
| Compute hours | 10 / workspace / month | A weekly run of ~10–15 min uses about 1 h/month |
| Memory | 2 GB, **including `pip install`** | Why the notebook-only packages were moved into the `eda` dependency group — `pip install -e .` ignores dependency groups, so they never reach the container |
| Max run time | 24 h | Not a factor here |

If a run dies with a memory error, the likely cause is `run_experiments.py` training all four
models concurrently. Serialize them:

```python
from prefect.task_runners import ThreadPoolTaskRunner

@flow(name="run_experiments", task_runner=ThreadPoolTaskRunner(max_workers=1))
```

### 3.5 What made this work

Four changes, all inert when running locally:

| Blocker | Fix |
|---------|-----|
| `src/churn` never installed; `requirements.txt` was stale | `requirements.txt` is now `-e .`, so the container installs the project from `pyproject.toml`. **Editable** matters: `mlflow_tracking_uri()` resolves paths relative to `__file__`, which a `site-packages` install would break. |
| `load_kaggle_credentials` required `kaggle/kaggle.json` | Already-set `KAGGLE_USERNAME` / `KAGGLE_KEY` now win, and the file is the fallback. |
| MLflow wrote to a container-local SQLite file | `mlflow_tracking_uri()` returns `$MLFLOW_TRACKING_URI` when set, else the local SQLite path. |
| Predictions written to a disposable `data/predictions/` | `run_batch_prediction.py` logs that batch's parquet + CSVs and its summary metrics to MLflow, which is remote. |

## 4. Switching between backends

The profile is the only switch. It persists until changed, across terminals and reboots.

| Goal | Command |
|------|---------|
| Go to Cloud | `uv run prefect profile use prefect-cloud` |
| Go back to local | `uv run prefect profile use local` |
| Check where you are | `uv run prefect config view \| grep API_URL` |
| No backend at all (ephemeral) | `uv run prefect config unset PREFECT_API_URL` |

Rules of thumb:

- **Every terminal reads the same profile**, so switching in one affects all of them. A
  `serve()` process started *before* a switch keeps talking to the old backend until restarted.
- Restart `scripts/run_weekly.py` after switching, otherwise the deployment stays registered on
  the old backend and `prefect deployment run` reports *not found*.
- Flow-run history does **not** move between backends. Local runs stay in `~/.prefect/prefect.db`,
  Cloud runs stay in the workspace.
- MLflow is unaffected by the Prefect profile — it follows `MLFLOW_TRACKING_URI`, and falls back
  to the local `mlflow.db` when that is unset (§3.1a).
- `scripts/clean_slate.sh` only clears **local** Prefect state; Cloud run history must be
  deleted from the Cloud UI.

## 5. Assets — the lineage graph

Prefect **assets** are the things the pipeline produces, as opposed to the tasks that produce
them. Declaring them turns three separate flows into one connected graph under *Assets* in the
Cloud UI, showing what each run touched and where a stale file came from.

Assets are a **Prefect Cloud feature**; with a local backend the decorators behave as ordinary
tasks and nothing is recorded.

```
kaggle://blastchar/telco-customer-churn        (referenced, never written)
  └─ file://data/raw/extraction_metadata_log.json
       └─ file://data/processed/churn_cleaned.csv  +  transformation_log.json
            ├─ mlflow://telco-churn-logistic-regression
            ├─ mlflow://telco-churn-random-forest
            ├─ mlflow://telco-churn-xgboost
            ├─ mlflow://telco-churn-segmentation
            ├─ mlflow://telco-churn-model              (the primary champion)
            └─ file://data/processed/batch_customers.parquet
                 └─ file://data/predictions/latest_predictions.parquet
                      └─ file://data/predictions/latest_model_comparison.csv
```

Keys live in one place, `src/churn/assets.py`, for the same reason column names live in
`features/schema.py`: a typo doesn't fail, it silently splits one node into two.

Each materialization carries metadata visible on the asset: row and column counts on the cleaned
CSV, the registered version plus every scalar metric (`roc_auc`, `f1`, …) on each model, and the
batch ID with predicted-churner count on the predictions.

Two edges cannot be inferred from the task graph and are declared explicitly:

- **cleaned CSV → models**, because `run_experiments.py` reads the CSV with `pd.read_csv`
  rather than receiving it from an upstream task;
- **models → predictions**, because `load_mlflow_model` takes URI *strings*. This one is
  declared at the call site in `run_batch_prediction.py` using
  `save_predictions.with_options(asset_deps=...)`, built from the config so no model name is
  hardcoded. (`with_options` accepts `asset_deps` but not `assets` — the assets themselves are
  fixed at import time.)

The `file://` keys describe container-local paths, which is correct for lineage; the durable
copy of each output is the MLflow artifact from §3.5.

One trap worth knowing: `@materialize` **is** a task decorator. Stacking `@task` on top of it
wraps the materializing task and the asset is never recorded.

## Troubleshooting

**`Deployment '.../...' not found!`** — usually a backend mismatch: the deployment was served
against one backend and `prefect deployment run` is asking a different one. Run
`prefect config view | grep API_URL` in both terminals, then `prefect deployment ls` to see what
the active backend actually has. Remember `serve()` deployments vanish when the process stops.

**`Unauthorized` / `401`** — the stored API key expired. Re-run `uv run prefect cloud login`.

**Run stuck in `Pending` / `Scheduled` in Mode B** — the managed pool is provisioning, or the
pull step failed. Open the flow run in the Cloud UI and read the infrastructure logs.

**`ModuleNotFoundError: No module named 'churn'` in Mode B** — the pull step didn't install the
project. Check that `requirements.txt` still contains `-e .` and that `--with-requirements
requirements.txt` was passed to `deploy`.

**`FileNotFoundError: Kaggle credentials not found` in Mode B** — the `KAGGLE_USERNAME` /
`KAGGLE_KEY` secrets are missing or misnamed on the deployment. Re-run `deploy` with both
`--secret` flags.

**Models train but stage 3 can't resolve `models:/...@champion`** — either `MLFLOW_TRACKING_URI`
isn't reaching the container (stage 2 wrote to a SQLite file that no longer exists), or the
server doesn't support aliases. Check the DagsHub UI for the new versions, then §3.1b.

**No assets in the Cloud UI** — assets need a Cloud backend; a local profile records nothing.
Also confirm no `@task` sits above a `@materialize`.
