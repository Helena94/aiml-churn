# Switching between local and cloud

There are **two switches, and they are independent**. Most confusion about "am I running
locally or in the cloud?" comes from assuming there is only one.

| Switch | Controls | Set by | Read at |
|--------|----------|--------|---------|
| **Prefect profile** | Orchestration — where flow runs, logs and assets are recorded | `prefect profile use <name>` | `~/.prefect/profiles.toml` |
| **`MLFLOW_TRACKING_URI`** | Tracking — where experiments, metrics and the model registry live | shell `export` (or a deployment `--env`) | [`src/churn/flows/mlflow.py:20`](src/churn/flows/mlflow.py#L20) |

Nothing ties them together. You can run a flow orchestrated by Prefect Cloud that writes its
models to the SQLite file on your laptop, or a flow with no Prefect backend at all that writes
to a hosted MLflow server. Both are useful — see [the four combinations](#the-four-combinations).

## Where am I right now?

Run this before anything that matters. It is the whole diagnostic:

```bash
uv run prefect config view | grep API_URL                    # orchestration
echo "${MLFLOW_TRACKING_URI:-<unset — local mlflow.db>}"     # tracking
```

Reading the output:

| Line | Local | Cloud |
|------|-------|-------|
| `PREFECT_API_URL` | `http://127.0.0.1:4200/api`, or absent (ephemeral) | `https://api.prefect.cloud/api/accounts/<id>/workspaces/<id>` |
| `MLFLOW_TRACKING_URI` | `<unset>` | `https://dagshub.com/<user>/<repo>.mlflow` |

The second line is the one the old docs never told you to check, and it is the one that bites:
a leftover `export MLFLOW_TRACKING_URI` from a deploy-verification session makes every
subsequent "local" run write to DagsHub, with no visible sign in the Prefect output.

## Flipping each switch

### Prefect profile — persistent, global

```bash
uv run prefect profile ls                     # * marks the active one
uv run prefect profile use local              # → local server at 127.0.0.1:4200
uv run prefect profile use prefect-cloud      # → the Cloud workspace
uv run prefect config unset PREFECT_API_URL   # → no backend at all (ephemeral)
```

This project's `~/.prefect/profiles.toml` already defines both `local` and `prefect-cloud`.
The names are just labels — what matters is the API URL each one sets. If a profile is missing,
create it with `prefect profile create <name>` then `prefect config set PREFECT_API_URL=...`,
or run `uv run prefect cloud login` to have the Cloud one written for you.

The setting **persists across terminals and reboots** until you change it again.

### `MLFLOW_TRACKING_URI` — per-shell, not persistent

```bash
# → remote
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>

# → back to the project-local SQLite store
unset MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
```

Unset, `mlflow_tracking_uri()` falls back to `sqlite:///<project>/mlflow.db`, which is why local
runs need no configuration at all. A **new terminal does not inherit the export** — the opposite
failure mode from the Prefect profile, and worth keeping straight:

- Prefect profile: change it once, every terminal is affected, including ones already open.
- `MLFLOW_TRACKING_URI`: set it in one terminal, no other terminal knows.

`configs/batch_prediction.yml` has `tracking_uri: null`, so stage 3 inherits exactly the same
resolution rather than carrying its own setting.

## The four combinations

| Prefect | MLflow | What it is | When to use it |
|---------|--------|-----------|----------------|
| local | local | Fully offline development | The default. [RUN_LOCAL.md](RUN_LOCAL.md) |
| Cloud | local | **Mode A** — flow executes on your machine, Cloud schedules and records it | Cloud UI, run history and the asset graph without giving up the local data and registry. [RUN_PREFECT_CLOUD.md §2](RUN_PREFECT_CLOUD.md#2-mode-a--execute-locally-orchestrate-in-cloud-recommended) |
| local | remote | **Pre-deploy verification** | Prove the remote tracking server works before spending Cloud compute. [RUN_PREFECT_CLOUD.md §3.1b](RUN_PREFECT_CLOUD.md#31-one-time-setup) |
| Cloud | remote | **Mode B** — managed container, nothing of yours runs | The real deployment. [RUN_PREFECT_CLOUD.md §3](RUN_PREFECT_CLOUD.md#3-mode-b--execute-on-clouds-managed-infrastructure) |

The two mixed rows are not mistakes. Mode A is the recommended way to use Cloud, and the
local-Prefect/remote-MLflow row is the cheapest bug-catching step in the whole deployment
process — it is what turns an 8-minute Cloud failure into a 20-second one on your laptop.

## Rules and traps

**A running `serve()` keeps its old backend.** `scripts/run_weekly.py` connects when it starts.
Switch profiles afterwards and it stays attached to the previous backend — `prefect deployment
run` then reports *not found* because your CLI is asking a different server. Restart the serving
process after every switch.

**Deployment names differ by how you deployed.** `serve()` registers
`weekly-churn-analysis/weekly-churn-analysis` (from `@flow(name=...)`); `prefect-cloud deploy`
registers `weekly_pipeline/weekly-churn-analysis` (from the entrypoint *function* name). Same
file, two addresses. `prefect deployment ls` / `uvx prefect-cloud ls` print the truth.

**Run history does not migrate.** Local runs stay in `~/.prefect/prefect.db`; Cloud runs stay in
the workspace. Switching does not move or merge them, and neither backend can see the other's.

**Assets are Cloud-only.** With a local backend the `@materialize` decorators behave as ordinary
tasks and nothing is recorded — the lineage graph is empty by design, not broken.

**`clean_slate.sh` only cleans local state.** [`scripts/clean_slate.sh:27`](scripts/clean_slate.sh#L27)
deletes `./mlflow.db` unconditionally; it never reads `MLFLOW_TRACKING_URI`. With that variable
set, the script deletes a database your runs were not using while DagsHub keeps every experiment
and registered model — a clean slate that is not one. Remote state must be deleted from the
DagsHub UI, and Cloud run history from the Prefect Cloud UI.

**Mixing registries mid-pipeline breaks stage 3.** `run_experiments.py` registers models wherever
`MLFLOW_TRACKING_URI` pointed at the time; `run_batch_prediction.py` resolves
`models:/...@champion` wherever it points *now*. Train remote and score local (or the reverse)
and the aliases will not resolve. Keep the variable constant across stages 2 and 3.

## Cheatsheet

```bash
# → fully local
uv run prefect profile use local
unset MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD

# → Mode A: Cloud orchestration, local data and registry
uv run prefect profile use prefect-cloud
unset MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD

# → pre-deploy verification: local orchestration, remote registry
uv run prefect profile use local
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user> MLFLOW_TRACKING_PASSWORD=<token>

# → confirm, always
uv run prefect config view | grep API_URL
echo "${MLFLOW_TRACKING_URI:-<unset — local mlflow.db>}"
```

Mode B needs no local switching: the container gets both settings from the deployment's `--env`
and `--secret` flags, independent of whatever your shell is doing.
