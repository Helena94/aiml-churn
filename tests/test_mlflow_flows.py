"""Unit tests for champion promotion.

`register_champion` is the only place that decides what `models:/<name>@champion` resolves
to, so it is exercised against a stub registry rather than a live MLflow server.
"""

from types import SimpleNamespace

import pytest

from churn.flows import mlflow as flows_mlflow
from churn.flows.mlflow import SELECTION_METRIC, register_champion

REGISTRY = "telco-churn-xgboost"


class StubClient:
    """Stands in for MlflowClient: runs keyed by id, aliases keyed by (name, alias)."""

    def __init__(self, runs, aliases):
        self.runs = runs  # run_id -> {metric: value}
        self.aliases = aliases  # (name, alias) -> SimpleNamespace(version, run_id)
        self.alias_calls = []

    def get_run(self, run_id):
        if run_id not in self.runs:
            raise RuntimeError(f"no run {run_id}")
        return SimpleNamespace(data=SimpleNamespace(metrics=self.runs[run_id]))

    def get_model_version_by_alias(self, name, alias):
        if (name, alias) not in self.aliases:
            raise RuntimeError(f"no alias {alias} on {name}")
        return self.aliases[(name, alias)]

    def set_registered_model_alias(self, name, alias, version):
        self.alias_calls.append((name, alias, int(version)))
        self.aliases[(name, alias)] = SimpleNamespace(version=int(version), run_id=None)


@pytest.fixture
def registry(monkeypatch):
    """Patches out the tracking server; returns a handle for arranging and asserting state."""
    state = SimpleNamespace(client=None, registered=[])

    def build(runs, aliases=None):
        state.client = StubClient(runs, dict(aliases or {}))
        return state

    def fake_register_model(uri, name):
        state.registered.append((uri, name))
        return SimpleNamespace(version=len(state.registered) + 1)

    monkeypatch.setattr(flows_mlflow.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(flows_mlflow.mlflow, "register_model", fake_register_model)
    monkeypatch.setattr(flows_mlflow, "MlflowClient", lambda *a, **kw: state.client)
    state.arrange = build
    return state


def test_registers_when_no_champion_exists(registry):
    """First run for a model: nothing to beat, so it is registered and aliased."""
    registry.arrange({"run-a": {SELECTION_METRIC: 0.80}})

    version, promoted = register_champion("run-a", "xgboost", REGISTRY)

    assert (version, promoted) == (2, True)
    assert registry.registered == [("runs:/run-a/model-xgboost", REGISTRY)]
    assert registry.client.alias_calls == [(REGISTRY, "champion", 2)]


def test_promotes_when_challenger_beats_incumbent(registry):
    registry.arrange(
        runs={"old": {SELECTION_METRIC: 0.80}, "new": {SELECTION_METRIC: 0.85}},
        aliases={(REGISTRY, "champion"): SimpleNamespace(version=1, run_id="old")},
    )

    version, promoted = register_champion("new", "xgboost", REGISTRY)

    assert promoted is True
    assert registry.client.alias_calls == [(REGISTRY, "champion", version)]


def test_keeps_incumbent_when_challenger_loses(registry):
    """The losing run must not enter the registry at all, and the alias must not move."""
    registry.arrange(
        runs={"old": {SELECTION_METRIC: 0.85}, "new": {SELECTION_METRIC: 0.80}},
        aliases={(REGISTRY, "champion"): SimpleNamespace(version=1, run_id="old")},
    )

    version, promoted = register_champion("new", "xgboost", REGISTRY)

    assert (version, promoted) == (1, False)
    assert registry.registered == []
    assert registry.client.alias_calls == []


def test_tie_keeps_incumbent(registry):
    """Re-running training on unchanged data is a no-op rather than a version churn."""
    registry.arrange(
        runs={"old": {SELECTION_METRIC: 0.85}, "new": {SELECTION_METRIC: 0.85}},
        aliases={(REGISTRY, "champion"): SimpleNamespace(version=3, run_id="old")},
    )

    assert register_champion("new", "xgboost", REGISTRY) == (3, False)
    assert registry.registered == []


def test_unscored_challenger_loses_to_incumbent(registry):
    """A run missing the selection metric can't be shown to be better, so it isn't promoted."""
    registry.arrange(
        runs={"old": {SELECTION_METRIC: 0.85}, "new": {"roc_auc": 0.99}},
        aliases={(REGISTRY, "champion"): SimpleNamespace(version=1, run_id="old")},
    )

    assert register_champion("new", "xgboost", REGISTRY) == (1, False)
    assert registry.registered == []


def test_unscored_challenger_is_promoted_when_registry_is_empty(registry):
    """Nothing to compare against beats nothing at all — the alias has to point somewhere."""
    registry.arrange({"new": {}})

    _, promoted = register_champion("new", "xgboost", REGISTRY)

    assert promoted is True


def test_promotes_over_incumbent_whose_run_is_gone(registry):
    """A deleted incumbent run leaves the alias unmeasurable; a scored challenger takes it."""
    registry.arrange(
        runs={"new": {SELECTION_METRIC: 0.70}},
        aliases={(REGISTRY, "champion"): SimpleNamespace(version=1, run_id="deleted")},
    )

    _, promoted = register_champion("new", "xgboost", REGISTRY)

    assert promoted is True


def test_ranks_on_the_requested_metric(registry):
    """The metric is a parameter — kmeans and the classifiers share one code path."""
    registry.arrange(
        runs={"old": {"silhouette": 0.4}, "new": {"silhouette": 0.5, SELECTION_METRIC: 0.1}},
        aliases={(REGISTRY, "champion"): SimpleNamespace(version=1, run_id="old")},
    )

    _, promoted = register_champion("new", "kmeans", REGISTRY, metric="silhouette")

    assert promoted is True
