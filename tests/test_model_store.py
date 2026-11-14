"""Rechargement à chaud : l'alias est relu périodiquement, un échec de relecture ne coupe pas le service."""
import pytest

from api.model_store import LoadedModel, ModelStore, ModelUnavailable


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def store(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("api.model_store.time.monotonic", clock)
    s = ModelStore("bikeshare", "champion", ["hr"], reload_interval=10)
    s.clock, s.versions, s.calls = clock, ["1"], 0

    def fake_load():
        s.calls += 1
        value = s.versions[0]
        if value == "down":
            raise ModelUnavailable("MLflow injoignable")
        return LoadedModel("bikeshare", value, "run", object(), 0.0)
    monkeypatch.setattr(s, "_load", fake_load)
    return s


def test_alias_is_not_reread_before_interval(store):
    assert store.current().version == "1"
    store.versions[0] = "2"                # le champion change dans le registre
    store.clock.now += 5
    assert store.current().version == "1" and store.calls == 1


def test_new_champion_is_picked_up_after_interval(store):
    store.current()
    store.versions[0] = "2"
    store.clock.now += 11
    assert store.current().version == "2"


def test_failed_refresh_keeps_serving_current_model(store):
    store.current()
    store.versions[0] = "down"
    store.clock.now += 11
    assert store.current().version == "1"


def test_no_model_at_all_is_unavailable(store):
    store.versions[0] = "down"
    with pytest.raises(ModelUnavailable):
        store.current()
