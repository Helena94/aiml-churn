# Prefect Cloud deployment — what failed, and why it all passed locally

Record of the debugging session that took `weekly_pipeline/weekly-churn-analysis` from failing on
every Cloud run to a green end-to-end run. Written because the interesting part is not the four
bugs themselves, but that **every one of them passed locally and failed only in Prefect Cloud's
managed container**.

Final state: nine Cloud runs, five commits, one green pipeline.

| | |
|---|---|
| Green run | `optimal-reindeer`, 2026-08-06 16:57:45 → 17:06:41 (~9 min) |
| Result | 1,409 customers scored, 284 predicted churners, Low 997 / Medium 327 / High 85 |
| Models | 4 registered on DagsHub, all loaded back through `models:/…@champion` |

---

## 1. Why local success predicted nothing

The local environment and the Cloud container differ on six axes at once. Each bug below lives in
exactly one of these gaps.

| | Local | Prefect Cloud managed container |
|---|---|---|
| Python | 3.13 (`.venv`) | **3.12** (`prefecthq/prefect-client:3-python3.12`) |
| Install source | `uv sync` from `uv.lock` — exact versions | `uv pip install -r requirements.txt`, which was just `-e .` → **resolved `pyproject.toml` ranges fresh on every run** |
| `churn` importable because | editable install in the active `.venv` | must come from the fresh clone |
| MLflow backend | `sqlite:///mlflow.db` | **DagsHub** (`MLFLOW_TRACKING_URI`) |
| Filesystem | persistent | disposable |
| Credentials | `kaggle/kaggle.json` | `KAGGLE_*` secrets |

The single highest-value lesson: **`uv.lock` never reached the container.** `requirements.txt`
contained only `-e .`, so the container re-resolved the dependency ranges in `pyproject.toml` on
every run and was free to install newer releases than the pipeline was ever tested against. Two of
the four bugs are direct consequences.

---

## 2. Failure 0 — wrong deployment name

```
prefect_cloud.utilities.exception.ObjectNotFound
404 .../deployments/name/weekly-churn-analysis/weekly-churn-analysis
```

`RUN_PREFECT_CLOUD.md` §3.3 says to run `weekly-churn-analysis/weekly-churn-analysis`. The real
name is **`weekly_pipeline/weekly-churn-analysis`**.

A Prefect deployment is addressed `<flow-name>/<deployment-name>`. `scripts/run_weekly.py` declares
`@flow(name="weekly-churn-analysis")`, so the local `serve()` path in `__main__` does produce
`weekly-churn-analysis/weekly-churn-analysis` — the doc is correct for Mode A. But
`prefect-cloud deploy` takes the flow name from the **entrypoint function** (`…:weekly_pipeline`)
rather than from the decorator, giving `weekly_pipeline/weekly-churn-analysis`.

Confirmed with `uvx prefect-cloud ls`, and by `prefect.flow.name: 'weekly_pipeline'` in the
deployment labels. Same repository, same file, two different names depending on how it is deployed.

---

## 3. Failure 1 — `ModuleNotFoundError: No module named 'churn'`

Runs affected: `teal-dogfish`, `daffy-robin`, `cordial-raven`, `gifted-yak`, `merciful-oxpecker`.
Each crashed ~25 s in, before any pipeline stage.

```
File "/opt/prefect/aiml-churn-main/scripts/run_weekly.py", line 13, in <module>
  from run_batch_prediction import batch_prediction
File ".../scripts/run_batch_prediction.py", line 12, in <module>
  from churn.assets import BATCH_INPUT_ASSET, model_asset
ModuleNotFoundError: No module named 'churn'
```

The pull step `uv pip install -r requirements.txt` reported success and ran for 11–14 s (real work,
not an instant failure), yet `churn` was not importable by `/usr/local/lib/python3.12`, the
interpreter that executes the flow.

**Ruled out by evidence, not assumption:**

| Hypothesis | How it was eliminated |
|---|---|
| `requirements.txt` stale or unpushed | committed, and `git diff origin/main` empty |
| `src/` not committed | 26 files tracked under `src/`, no `.gitignore` match |
| Broken packaging / `src`-layout wrong | `git archive main` → clean clone → fresh venv → `uv pip install -e .` → `import churn` **worked** |
| `uv` refusing to install without a venv | `UV_SYSTEM_PYTHON=1` set on the deployment, verified present via `prefect deployment inspect`, **no change** |
| `uv` targeting the wrong interpreter | `UV_PYTHON=/usr/local/bin/python3.12` set and verified present, **no change** |

**Why the container could not be inspected:** Prefect does not surface pull-step subprocess output
in the flow-run logs. A diagnostic pull step that printed the interpreter, site-packages contents
and import result produced nothing visible — even with `stream_output: true`, and even after
merging stderr into stdout (`uv` writes to stderr). That observability gap is why this took five
runs.

**Fix** — `d5908c4`, `scripts/run_weekly.py`:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```

Three lines before the sibling imports. Inert locally, where the editable install already covers it.

**Honest limitation:** this is a workaround, not a root cause. *Why* the container's `-e .` install
never became importable was never determined, because the container could not be inspected. The fix
removes the dependency on that install rather than repairing it. The dependency install itself was
fine all along — proven when the very next run imported `pandas`, `scikit-learn` and `mlflow`
without complaint.

**Verified before deploying:** the path resolves to the real `src/`; a system Python with no `churn`
installed anywhere imports it with only that insert; and Prefect's own
`import_object("scripts/run_weekly.py:weekly_pipeline")` — the exact call that was crashing — loads
the flow.

---

## 4. Failure 2 — skops refuses to serialize a `SearchCV`

Run `screeching-goldfish` cleared the import, ran ETL, trained for minutes, then:

```
MlflowException: The saved sklearn model references untrusted types.
Untrusted types found: ['sklearn.metrics._ranking.roc_auc_score',
                       'sklearn.metrics._scorer._Scorer',
                       'sklearn.model_selection._split.StratifiedKFold']
```

Newer MLflow flips the default `serialization_format` of `mlflow.sklearn.log_model` from
`cloudpickle` to **`skops`**. skops refuses untrusted types, and a `GridSearchCV` carries exactly
those — a scorer and a CV splitter. Local MLflow 3.9.0 still defaults to `cloudpickle`, so the call
had never failed here.

**Proof the versions differed**, rather than assuming: the container traceback puts `_save_model(`
at line 301; the local file has it at line 295. Same file, different release.

**Fix** — `65c0615`, `src/churn/flows/mlflow.py`: state the format explicitly instead of inheriting
a default that changes underneath the project.

Verified by round-tripping a real `GridSearchCV` (roc_auc scorer + `StratifiedKFold`) through
`log_model` → `load_model` and asserting identical predictions.

---

## 5. Failure 3 — dependency drift, fixed at the source

Failure 2 was a symptom of the structural problem in §1: the container resolved ranges instead of
the lock.

- `8cf5bf1` — pinned `mlflow==3.9.0` in `pyproject.toml`.
- `8a7c59a` — replaced `requirements.txt` (previously the single line `-e .`) with the **exported
  lock**: 536 lines pinning all direct *and* transitive dependencies.

```bash
uv lock && uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
```

`-e .` stays first and stays editable, because `churn/flows/mlflow.py` resolves paths through
`Path(__file__).parents[3]`, which a `site-packages` install would break. The `dev` and `eda` groups
are excluded, so jupyter and ydata-profiling still never reach the 2 GB container.

**Verified against Python 3.12** — the container's version, not the local 3.13 — by installing the
file into a clean 3.12 environment:

| | Container (3.12) | Local |
|---|---|---|
| mlflow | 3.9.0 | 3.9.0 |
| scikit-learn | 1.8.0 | 1.8.0 |
| xgboost | 3.1.3 | 3.1.3 |
| pandas | 2.3.3 | 2.3.3 |
| prefect | 3.6.13 | 3.6.13 |
| numpy / scipy | 2.3.5 / 1.16.3 | same |

---

## 6. Failure 4 — DagsHub does not implement MLflow 3's logged-models API

Runs `true-mongoose` and `adamant-dove` both trained successfully (~8.5 min each) and then died at
registration:

```
MlflowException: Unable to find a logged_model with artifact_path
model-logistic_regression under run aca08aac86404490af0e78fc66b24131
```

**A wrong diagnosis, and how it was caught.** This was first attributed to version drift, since a
local test of the same `runs:/…` registration passed. `adamant-dove` then failed *identically* with
`mlflow==3.9.0` pinned — disproving it. The local test had passed only because it used **SQLite**.
The container talks to **DagsHub**. The variable was never the MLflow version; it was the tracking
backend.

**The real mechanism**, from `mlflow/tracking/_model_registry/fluent.py:172-188`:

1. `register_model("runs:/<run_id>/<name>")` lists artifacts under that run path and looks for an
   `MLmodel` file.
2. If found → classic registration, supported by every MLflow-compatible server.
3. If **not** found → fall back to MLflow 3's **logged-models API**.

MLflow 3's `log_model(name=…)` creates a "logged model" entity stored *outside* the run's artifact
tree, so step 1 always fails and step 3 always runs. SQLite implements that API. **DagsHub does
not.**

Switching to `artifact_path=` is not an option — `mlflow/models/model.py:1112` is literally
`name = name or artifact_path`, so it is a pure alias with identical behaviour.

**Fix** — `cece685`, `src/churn/flows/mlflow.py`: save the model locally, then upload it under the
**run's own** artifact path, so registration takes the classic branch and the unsupported API is
never consulted.

```python
artifact_path = "model-{}".format(model_name)
with tempfile.TemporaryDirectory() as tmp:
    local_model_path = Path(tmp) / artifact_path
    mlflow.sklearn.save_model(model, str(local_model_path),
                              serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    mlflow.log_artifacts(str(local_model_path), artifact_path=artifact_path)
```

**Verified by simulating DagsHub rather than trusting SQLite:** `_get_logged_models_from_run` was
stubbed to raise if called, then the real `register_champion` was run against it. Registration
succeeded, the `@champion` alias was set, and the model loaded back with identical predictions —
without ever touching the unsupported API.

---

## 7. Answered: DagsHub *does* support `@champion` aliases

`RUN_PREFECT_CLOUD.md` §3.1b flagged alias support as "the one real unknown", with a fallback to
`models:/<name>/latest`. Stage 3 of `optimal-reindeer` loaded all four models through
`models:/…@champion` URIs and scored 1,409 customers with them.

**`configs/batch_prediction.yml` needs no change.** The fallback stays unused.

---

## 8. Run timeline

| Run | Outcome |
|---|---|
| `teal-dogfish`, `daffy-robin`, `cordial-raven` | `ModuleNotFoundError: churn` |
| `gifted-yak` | same, after `UV_PYTHON` — hypothesis disproved |
| `merciful-oxpecker` | same; diagnostic pull step produced no visible output |
| `screeching-goldfish` | **import fixed**; ETL + training ran; died on skops |
| `true-mongoose` | skops fixed; died on `register_model` |
| `adamant-dove` | MLflow pinned; died identically — drift theory disproved |
| `optimal-reindeer` | **Completed**, all three stages |

## 9. Commits

| Commit | Change |
|---|---|
| `d5908c4` | Cloud entrypoint imports `churn` without the editable install |
| `65c0615` | Pin the sklearn serialization format when logging models |
| `8cf5bf1` | Pin MLflow to the verified version |
| `8a7c59a` | Pin every dependency for the deployed container |
| `cece685` | Log models under the run artifact path so registration works on DagsHub |

---

## 10. What would have caught these sooner

1. **Point the local verification at the real backend.** `RUN_PREFECT_CLOUD.md` §3.1b exists exactly
   for this. Run against `MLFLOW_TRACKING_URI=https://dagshub.com/…` — not the local `mlflow.db`.
   Failure 4 is a pure client-server compatibility bug and would have appeared in seconds on a
   laptop instead of after 8.5 minutes of Cloud training.
2. **Ship the lock to the container.** Failures 2 and 3 both come from `requirements.txt` being
   `-e .`. Now closed permanently.
3. **Test on the container's Python.** Local is 3.13, the container is 3.12.
4. **Assume pull-step output is invisible.** Anything needing verification must be provable *before*
   deploying — a clean `git archive` clone plus a throwaway venv reproduces the container closely
   enough for most questions, and costs no Cloud compute.

## 11. Still open

- **Why the container's `-e .` never became importable** was never root-caused (§3), only worked
  around. It cannot be diagnosed without shell access to the container or visible pull-step output.
- **`RUN_PREFECT_CLOUD.md` is stale**: §3.3 has the wrong deployment name (§2 here), and §3.5's
  "What made this work" table predates all five commits above.
- **`UV_PYTHON` / `UV_SYSTEM_PYTHON`** were patched onto the deployment while diagnosing and are now
  inert. They are not in the documented `deploy` command, so they vanish on the next redeploy —
  harmless either way.
- **No schedule is set.** The pipeline runs only when triggered:
  ```bash
  uvx prefect-cloud run weekly_pipeline/weekly-churn-analysis
  uvx prefect-cloud schedule weekly_pipeline/weekly-churn-analysis "0 7 * * 1"   # when ready; UTC
  ```
- **Predictions do not survive the container.** `data/predictions/` is discarded with the run; the
  durable copies are MLflow artifacts under the `batch-<batch_id>` run on DagsHub.
