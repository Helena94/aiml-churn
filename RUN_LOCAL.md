# Running locally

Everything runs on your machine: Prefect orchestration, MLflow tracking, and the data.
No account and no internet needed (except the Kaggle download in step 1).

For Prefect Cloud see [RUN_PREFECT_CLOUD.md](RUN_PREFECT_CLOUD.md); for moving between the two
see [SWITCHING.md](SWITCHING.md).

## 0. Prerequisites

| Need | How |
|------|-----|
| Python ≥ 3.10 | any install |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Dependencies | `uv sync --group dev` (add `--group eda` to run the notebooks) |
| Kaggle API token | download from kaggle.com → Account → Create New API Token, save as `kaggle/kaggle.json` |

`kaggle/` and `data/` are gitignored, so the token and the datasets never leave your machine.

Nothing here needs environment variables. MLflow writes to the project-local `mlflow.db` unless
`MLFLOW_TRACKING_URI` is set, and Kaggle credentials come from `kaggle/kaggle.json` unless
`KAGGLE_USERNAME` / `KAGGLE_KEY` are already in the environment — the remote paths are only used
by the managed deployment in [RUN_PREFECT_CLOUD.md](RUN_PREFECT_CLOUD.md).

## 1. Confirm you are actually local

The most common source of confusion in this project, and it has **two** independent causes.
Check both — one command each:

```bash
uv run prefect config view | grep API_URL                    # orchestration
echo "${MLFLOW_TRACKING_URI:-<unset — local mlflow.db>}"     # tracking
```

For local work you want:

```
PREFECT_API_URL='http://127.0.0.1:4200/api'      (or no API_URL line at all — ephemeral)
MLFLOW_TRACKING_URI: <unset — local mlflow.db>
```

**If the first line shows an `api.prefect.cloud` URL**, your flow runs are being recorded in a
Cloud workspace. Switch back with `uv run prefect profile use local`. Check the profile exists
first — `uv run prefect profile ls` prints them, and `local` is just a label; what matters is
the API URL it sets. To run with no backend at all, `uv run prefect config unset PREFECT_API_URL`.

**If the second line shows a URL**, every model you train is being registered on a *remote*
server, not in `mlflow.db` — and nothing in the Prefect output will say so. This usually means a
leftover `export` from a pre-deploy verification session
([RUN_PREFECT_CLOUD.md §3.1b](RUN_PREFECT_CLOUD.md#31-one-time-setup)). Clear it:

```bash
unset MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
```

The two switches are unrelated and behave differently — the profile is persistent and global,
the environment variable is per-shell and invisible. [SWITCHING.md](SWITCHING.md) covers both.

## 2. Run the pipeline

The three stages are strictly sequential — each consumes the previous one's output.
**A Prefect server is not required for this**; the scripts run in Prefect's ephemeral mode.

```bash
uv run python scripts/run_etl.py                 # → data/processed/churn_cleaned.csv
uv run python scripts/run_experiments.py         # → MLflow runs + registered models in mlflow.db
uv run python scripts/run_batch_prediction.py    # → data/predictions/*.parquet + comparison csv
```

(`uv run` uses the project venv without activating it. Plain `python` works only inside an
activated `.venv`.)

Skipping a step fails the next: without step 1 training has no data, without step 2 the
`models:/...@champion` URIs don't resolve.

## 3. See the results

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

Experiment runs, metrics, and the model registry (`telco-churn-model@champion` and the
per-model aliases). Predictions are plain files under `data/predictions/`.

Note the `--backend-store-uri` is spelled out, so this UI always shows the local `mlflow.db`
regardless of `MLFLOW_TRACKING_URI`. If step 2 ran with that variable set, its runs are on the
remote server and this UI will look empty — see section 1.

## 4. Optional: the local Prefect server + UI

Only needed if you want flow-run history and the schedule. Leave it running in its own
terminal.

```bash
uv run prefect server start                                       # terminal A
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

UI at http://127.0.0.1:4200. Re-running the scripts from step 2 now records their runs there.

## 5. Optional: the weekly schedule

`scripts/run_weekly.py` wraps steps 1–3 in one parent flow (`weekly-churn-analysis`) and
serves it on cron `0 9 * * 1` (Mondays 09:00, Europe/Belgrade), `limit=1` so runs never
overlap.

```bash
uv run prefect server start                # terminal A — keep running
uv run python scripts/run_weekly.py        # terminal B — keep running, this IS the deployment
```

**`serve()` is a foreground process.** The deployment exists only while terminal B is alive;
close it and the deployment disappears. It runs its own runner, so no work pool and no worker
are needed.

Trigger a run immediately instead of waiting for Monday, from a third terminal:

```bash
uv run prefect deployment ls                                        # confirm the name first
uv run prefect deployment run 'weekly-churn-analysis/weekly-churn-analysis'
```

The command returns immediately; the run is picked up by terminal B, which prints the logs.

Because the Kaggle dataset is static, the schedule demonstrates Prefect orchestration — it is
not evidence that weekly retraining improves the models.

## Troubleshooting

**`Deployment 'weekly-churn-analysis/weekly-churn-analysis' not found!`** — one of three things:

1. `scripts/run_weekly.py` is not running. Deployments from `serve()` live only as long as the
   process (section 5).
2. Your CLI is on a different backend than the one that served the deployment. Compare
   `prefect config view | grep API_URL` in both terminals (section 1).
3. The name really is different. `prefect deployment ls` prints the truth. A deployment created
   with `prefect-cloud deploy` is registered under the entrypoint *function* name,
   `weekly_pipeline/weekly-churn-analysis`, not the flow's `@flow(name=...)` — the address above
   is the `serve()` one and applies here.

**`FileNotFoundError: Kaggle credentials not found`** — neither `KAGGLE_USERNAME` +
`KAGGLE_KEY` in the environment nor `kaggle/kaggle.json` on disk. Either one works; env vars win.

**MLflow model URI does not resolve** — `run_experiments.py` has not run against this
`mlflow.db`, or `clean_slate.sh` wiped the registry. Re-run step 2. If step 2 *did* run, check
`MLFLOW_TRACKING_URI` was the same for both stages — train remote and score local (or the
reverse) and the `@champion` aliases resolve against the wrong registry (section 1).

## Reset

```bash
scripts/clean_slate.sh        # -y to skip the confirmation prompt
```

Kills running mlflow/prefect processes, then deletes the MLflow store (`mlflow.db`, `mlruns/`,
`mlartifacts/`, every registered model and `@champion` alias), the prediction files, and the
local Prefect run history. Keeps source, `configs/`, `data/raw/`, and `data/processed/` — so
recovery is just steps 2 and 3 of the run order again.

**It only cleans local state.** [`scripts/clean_slate.sh:27`](scripts/clean_slate.sh#L27) deletes
`./mlflow.db` unconditionally and never reads `MLFLOW_TRACKING_URI`. With that variable set the
script wipes a database your runs were not using, while the remote server keeps every experiment
and registered model — a clean slate that is not one. Remote state has to be deleted from that
server's own UI, and Cloud run history from the Prefect Cloud UI.
