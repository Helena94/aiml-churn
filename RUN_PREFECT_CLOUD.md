# Running against Prefect Cloud

Same code, same commands — only the orchestration backend changes. Prefect Cloud gives a
hosted UI, run history, the asset lineage graph, and the schedule; the machine that actually
executes the flow depends on which of the two modes below you use.

For the fully local setup see [RUN_LOCAL.md](RUN_LOCAL.md). For the mechanics of moving between
the two — and the second switch that has nothing to do with Prefect — see
[SWITCHING.md](SWITCHING.md).

## 0. Prerequisites

Everything from [RUN_LOCAL.md §0](RUN_LOCAL.md#0-prerequisites) (uv, `uv sync --group dev`,
`kaggle/kaggle.json`), plus a free Prefect Cloud account with a workspace.

Mode B additionally needs the repo pushed to GitHub and a remote MLflow server (§3.1).

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

`serve()` registers the deployment in the Cloud workspace on cron `0 9 * * 1`
(`Europe/Belgrade`, `limit=1` so runs never overlap) and runs its own runner — **no work pool,
no worker**. The deployment exists only while this process is alive; close the terminal and it
disappears.

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
| Source | clones the GitHub repo, then `uv pip install -r requirements.txt` (a fully pinned lock export — §3.5) |
| Deployment name | **`weekly_pipeline/weekly-churn-analysis`** — not the Mode A name (§3.3) |
| Schedule | set separately with `prefect-cloud schedule`; none is set today |

### 3.0 How the container differs from your laptop

Six differences, and every Cloud-only bug this project has hit lived in exactly one of them.
Worth reading before deploying, because none of them show up in a local test:

| | Local | Managed container |
|---|---|---|
| Python | 3.13 (`.venv`) | **3.12** (`prefecthq/prefect-client:3-python3.12`) |
| Install source | `uv sync` from `uv.lock` | `uv pip install -r requirements.txt` |
| `churn` importable via | editable install in the active `.venv` | the fresh clone — see §3.5 |
| MLflow backend | `sqlite:///mlflow.db` | remote, via `MLFLOW_TRACKING_URI` |
| Filesystem | persistent | **disposable** |
| Credentials | `kaggle/kaggle.json` | `KAGGLE_*` secrets |

The disposable filesystem dictates the rest: `kaggle/`, `data/` and `*.db` are gitignored, so
credentials must arrive as secrets and anything worth keeping must be written somewhere remote.
All three stages run inside one flow run, so they do share a filesystem for the duration — only
the model registry and the outputs need to outlive it.

One more thing to plan around: **Prefect does not surface pull-step output in the flow-run
logs.** If the install step misbehaves you get no diagnostics, even with `stream_output: true`.
Anything that needs verifying must be provable *before* deploying. A `git archive main` clone
into a throwaway Python 3.12 venv reproduces the container closely enough for most questions and
costs no Cloud compute.

### 3.1 One-time setup

**a. A remote MLflow server.** [DagsHub](https://dagshub.com) is free, gives a hosted MLflow
tracking server *and* artifact storage in one signup, and connects to an existing GitHub repo.
Sign up, connect this repo, then take the tracking URL and a token from the repo's *Remote →
Experiments* tab.

Any MLflow server works — `MLFLOW_TRACKING_URI` is all the code reads
([`src/churn/flows/mlflow.py:20`](src/churn/flows/mlflow.py#L20)). Unset, it still falls back to
the local `mlflow.db`, so nothing about local runs changes.

**b. Verify it locally before spending any Cloud compute.** Do not skip this. It is the
cheapest step in the process and the one that pays for itself: a tracking-server incompatibility
surfaces here in seconds instead of after 8.5 minutes of Cloud training with no container logs
to read. That is not hypothetical — the registration failure in §3.6 (`cece685`) is a pure
client/server compatibility bug, and it cost two full Cloud runs precisely because the local
test that was run used SQLite rather than the real backend.

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
python scripts/run_experiments.py && python scripts/run_batch_prediction.py
```

Green, with the models visible in the DagsHub UI ⇒ continue. This exercises the two things a
non-reference MLflow server is most likely to get wrong: `register_model` and alias resolution.

Both are confirmed working on DagsHub. The `models:/<name>@champion` URIs in
`configs/batch_prediction.yml` need no change — an earlier version of this doc suggested a
`models:/<name>/latest` fallback in case aliases were unsupported; they are supported, and the
fallback is unnecessary.

**Unset the three variables afterwards**, or every later "local" run silently writes to DagsHub
— see [SWITCHING.md](SWITCHING.md#where-am-i-right-now).

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

**The deployment is not called what `--name` says.** A Prefect deployment is addressed
`<flow-name>/<deployment-name>`, and the two deployment paths derive the flow name differently:

| Deployed by | Flow name comes from | Full address |
|---|---|---|
| `serve()` (Mode A) | `@flow(name="weekly-churn-analysis")` | `weekly-churn-analysis/weekly-churn-analysis` |
| `prefect-cloud deploy` (Mode B) | the entrypoint **function**, `…:weekly_pipeline` | **`weekly_pipeline/weekly-churn-analysis`** |

Same file, same `--name`, two different addresses. Using the Mode A name here returns a bare
`404 ObjectNotFound`. Confirm with `uvx prefect-cloud ls` rather than trusting either.

```bash
uvx prefect-cloud run weekly_pipeline/weekly-churn-analysis
uvx prefect-cloud schedule weekly_pipeline/weekly-churn-analysis "0 7 * * 1"
```

Verify the manual run first — a broken schedule burns the free compute quota every week and
fills the run history with failures. **No schedule is currently set**; the pipeline runs only
when triggered.

`prefect-cloud schedule` takes **no timezone flag**, so the cron is UTC. `0 7 * * 1` is Monday
09:00 in Belgrade during summer (CEST) and 08:00 in winter (CET); the `serve()` schedule in
Mode A is timezone-aware and does not drift. Use the Cloud UI's schedule editor if you need a
timezone attached.

A successful run leaves: three green stages in Prefect Cloud, a new version of each model in
DagsHub, the batch outputs as MLflow artifacts under a `batch-<batch_id>` run, and a connected
asset graph (§5).

### 3.4 Free-tier limits worth knowing

| Limit | Value | Relevance |
|-------|-------|-----------|
| Compute hours | 10 / workspace / month | A weekly run of ~10–15 min uses about 1 h/month |
| Memory | 2 GB, **including the dependency install** | Why the notebook-only packages live in the `eda` group, which the requirements export excludes — they never reach the container |
| Max run time | 24 h | Not a factor here |

If a run dies with a memory error, the likely cause is `run_experiments.py` training all four
models concurrently. Serialize them:

```python
from prefect.task_runners import ThreadPoolTaskRunner

@flow(name="run_experiments", task_runner=ThreadPoolTaskRunner(max_workers=1))
```

### 3.5 Keeping the container reproducible

`requirements.txt` is **not** a hand-written file and is no longer the single line `-e .` it once
was. It is the exported lock — 547 lines pinning every direct *and* transitive dependency.

This matters because `uv.lock` never reaches the container: the pull step installs from
`requirements.txt`, so any unbounded version range would be re-resolved fresh on every run,
free to install releases the pipeline was never tested against. That drift broke MLflow twice.

**Regenerate it after any dependency change:**

```bash
uv lock && uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
```

Forget this and the container installs the *previous* dependency set while your laptop uses the
new one — the exact divergence the pinning exists to prevent. Commit the result.

Two properties of the file are deliberate and easy to break:

- **`-e .` stays first and stays editable.** `mlflow_tracking_uri()` resolves paths through
  `Path(__file__).resolve().parents[3]`, which a `site-packages` install would break.
- **`dev` and `eda` are excluded** (`--no-dev` plus the group not being a default), so pytest,
  jupyter and ydata-profiling never count against the 2 GB budget.

The container's import of `churn` does not depend on that editable install succeeding, though —
[`scripts/run_weekly.py:16`](scripts/run_weekly.py#L16) puts `src/` on `sys.path` directly:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```

Inert locally, where the editable install already covers it.

### 3.6 What made this work

Five commits took this deployment from failing on every run to green. All of them are inert
when running locally, which is exactly why none were caught before deploying:

| Commit | Blocker | Fix |
|---|---|---|
| `d5908c4` | `ModuleNotFoundError: No module named 'churn'` — the container's `-e .` install never became importable by the interpreter that executes the flow | Put `src/` on `sys.path` in the entrypoint (§3.5), removing the dependency on that install |
| `65c0615` | `MlflowException: references untrusted types` — newer MLflow flips `log_model`'s default serialization to **skops**, which refuses a `SearchCV` (it carries a scorer and a `StratifiedKFold`) | State `SERIALIZATION_FORMAT_CLOUDPICKLE` explicitly instead of inheriting a default that changes underneath the project |
| `8cf5bf1` | Same class of problem, at the source | Pin `mlflow==3.9.0` in `pyproject.toml` |
| `8a7c59a` | The container re-resolved dependency ranges on every run | Replace `requirements.txt` with the exported lock (§3.5) |
| `cece685` | `Unable to find a logged_model with artifact_path …` at registration — MLflow 3's `log_model(name=…)` stores the model *outside* the run's artifact tree, so `register_model` falls back to the logged-models API, which **DagsHub does not implement** | Save the model locally, then upload it under the run's own artifact path, so `register_model` takes its classic branch — supported by every MLflow-compatible server ([`src/churn/flows/mlflow.py:111-119`](src/churn/flows/mlflow.py#L111-L119)) |

Also relevant: `load_kaggle_credentials` prefers already-set `KAGGLE_USERNAME` / `KAGGLE_KEY`
and falls back to `kaggle/kaggle.json`, so the container's secrets work; and
`run_batch_prediction.py` logs each batch's parquet + CSVs and summary metrics to MLflow, which
is the only reason the outputs survive the disposable `data/predictions/`.

The full debugging record — nine Cloud runs, the hypotheses that were disproved, and how each
fix was verified — is in
[CLOUD_DEPLOYMENT_POSTMORTEM.md](CLOUD_DEPLOYMENT_POSTMORTEM.md).

## 4. Switching between backends

Moved to [SWITCHING.md](SWITCHING.md), which covers both switches — the Prefect profile *and*
`MLFLOW_TRACKING_URI` — the four combinations they produce, and the traps in each.

The one-line version: the profile is persistent and global, `MLFLOW_TRACKING_URI` is per-shell
and invisible, and they have nothing to do with each other.

```bash
uv run prefect config view | grep API_URL                    # orchestration
echo "${MLFLOW_TRACKING_URI:-<unset — local mlflow.db>}"     # tracking
```

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

Keys live in one place, [`src/churn/assets.py`](src/churn/assets.py), for the same reason column
names live in `features/schema.py`: a typo doesn't fail, it silently splits one node into two.

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
copy of each output is the MLflow artifact from §3.6.

One trap worth knowing: `@materialize` **is** a task decorator. Stacking `@task` on top of it
wraps the materializing task and the asset is never recorded.

## Troubleshooting

**`Deployment '.../...' not found!` / `404 ObjectNotFound`** — first check the name. Mode B is
`weekly_pipeline/weekly-churn-analysis`, Mode A is `weekly-churn-analysis/weekly-churn-analysis`
(§3.3). If the name is right, it is a backend mismatch: the deployment was served against one
backend and the command is asking a different one. Compare `prefect config view | grep API_URL`
in both terminals, then `prefect deployment ls` (or `uvx prefect-cloud ls`) to see what the
active backend actually has. Remember `serve()` deployments vanish when the process stops.

**`Unauthorized` / `401`** — the stored API key expired. Re-run `uv run prefect cloud login`.

**Run stuck in `Pending` / `Scheduled` in Mode B** — the managed pool is provisioning, or the
pull step failed. Open the flow run in the Cloud UI and read the infrastructure logs.

**`ModuleNotFoundError: No module named 'churn'` in Mode B** — the `sys.path` insert at the top
of [`scripts/run_weekly.py:16`](scripts/run_weekly.py#L16) is missing or was moved *below* the
sibling imports; it must run before them. Do not go looking at `requirements.txt` for this — the
container's `-e .` install was never what made the import work (§3.5, §3.6).

**`MlflowException: references untrusted types`** — the explicit `serialization_format` in
`src/churn/flows/mlflow.py` was dropped, or an unpinned MLflow reached the container. Check
§3.5's regenerate step ran after the last dependency change.

**`Unable to find a logged_model with artifact_path …`** — the model is being logged with
`log_model(name=…)` again instead of being saved and uploaded under the run's artifact path
(§3.6, `cece685`). Note this fails only against servers that don't implement MLflow 3's
logged-models API — a local SQLite test will pass and tell you nothing.

**`FileNotFoundError: Kaggle credentials not found` in Mode B** — the `KAGGLE_USERNAME` /
`KAGGLE_KEY` secrets are missing or misnamed on the deployment. Re-run `deploy` with both
`--secret` flags.

**Models train but stage 3 can't resolve `models:/...@champion`** — `MLFLOW_TRACKING_URI` isn't
reaching the container, so stage 2 wrote to a SQLite file that no longer exists. Check the
DagsHub UI for the new versions. (Alias support itself is not the problem — §3.1b.)

**No assets in the Cloud UI** — assets need a Cloud backend; a local profile records nothing.
Also confirm no `@task` sits above a `@materialize`.
