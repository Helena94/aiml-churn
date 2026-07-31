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

A deployment of this kind already exists in the workspace, created with `prefect-cloud deploy`:

| Field | Value |
|-------|-------|
| Deployment | `weekly_pipeline/weekly-churn-analysis` |
| Entrypoint | `scripts/run_weekly.py:weekly_pipeline` |
| Work pool | `default-work-pool` (type `prefect:managed` — Prefect runs it, no worker of yours) |
| Source | clones `github.com/Helena94/aiml-churn` @ `main`, then `uv pip install -r requirements.txt` |
| Schedule | none |

```bash
uv run prefect deployment run 'weekly_pipeline/weekly-churn-analysis'
```

Note the flow name is `weekly_pipeline` (the Python function name), not the `@flow(name=...)`
value `weekly-churn-analysis` used in Mode A. Both may appear in `prefect deployment ls` at the
same time; they are different deployments.

### Known limitation of Mode B

**This mode currently cannot complete the pipeline.** The managed container clones the public
repo, and `.gitignore` excludes `kaggle/`, `data/`, `*.json`, and `*.db`. So the container has:

- no `kaggle/kaggle.json` → `load_kaggle_credentials` raises `FileNotFoundError` and stage 1
  fails immediately (there is no environment-variable fallback in the code today);
- no `data/`, so nothing to fall back on;
- an ephemeral filesystem — the SQLite MLflow store is created inside the container and
  discarded when the run ends, so runs and registered models would not persist anyway.

Making Mode B work end to end needs credentials injected as Prefect secrets or job-variable
env vars, a code path that reads them, and a remote MLflow tracking server plus remote artifact
storage. That is deliberately out of scope for this project — see the *Scope and future work*
section of the [README](README.md#scope-and-future-work).

Use **Mode A** for a working demonstration of Cloud orchestration.

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
- MLflow is unaffected by the switch — it is always the local `mlflow.db` in the project root.
- `scripts/clean_slate.sh` only clears **local** Prefect state; Cloud run history must be
  deleted from the Cloud UI.

## Troubleshooting

**`Deployment '.../...' not found!`** — usually a backend mismatch: the deployment was served
against one backend and `prefect deployment run` is asking a different one. Run
`prefect config view | grep API_URL` in both terminals, then `prefect deployment ls` to see what
the active backend actually has. Remember `serve()` deployments vanish when the process stops.

**`Unauthorized` / `401`** — the stored API key expired. Re-run `uv run prefect cloud login`.

**Run stuck in `Pending` / `Scheduled` in Mode B** — the managed pool is provisioning, or the
pull step failed. Open the flow run in the Cloud UI and read the infrastructure logs.
