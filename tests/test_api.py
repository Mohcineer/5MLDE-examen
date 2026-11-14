"""API : validation, erreurs, version identifiée, traces et feedbacks (modèle factice, SQLite temporaire)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main
from api.model_store import LoadedModel, ModelUnavailable
from bikeshare.db import Database

VALID = {"season": 1, "yr": 1, "mnth": 6, "hr": 8, "holiday": 0, "weekday": 2, "workingday": 1,
         "weathersit": 1, "temp": 0.6, "atemp": 0.58, "hum": 0.5, "windspeed": 0.2}


class FakeModel:
    def predict(self, frame):
        return np.full(len(frame), 123.4)


class FakeStore:
    alias = "champion"

    def __init__(self, version="7"):
        self.loaded = LoadedModel("bikeshare", version, "run-abc", FakeModel(), 0.0)
        self.available = True
        self.features = list(VALID)

    def current(self):
        if not self.available:
            raise ModelUnavailable("MLflow injoignable")
        return self.loaded

    def frame(self, items):
        return main.ModelStore.frame(self, items)


@pytest.fixture
def client(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'api.db'}")
    database.init_schema()
    monkeypatch.setattr(main.State, "database", database)
    monkeypatch.setattr(main.State, "store", FakeStore())
    monkeypatch.setattr(main.State, "trace_failures", 0)
    with TestClient(main.app) as test_client:
        yield test_client


def rows(sql):
    return main.State.database.query(sql)


def test_predict_returns_version_and_is_traced(client):
    response = client.post("/predict", json=VALID, headers={"X-Campaign-Id": "t1"})
    body = response.json()
    assert response.status_code == 200
    assert (body["model_name"], body["model_version"], body["predicted_cnt"]) == ("bikeshare", "7", 123.4)
    request = rows("SELECT * FROM api_requests")[0]
    assert (request["endpoint"], request["http_status"], request["model_version"], request["campaign_id"]) == \
        ("/predict", 200, "7", "t1")
    assert request["processing_duration_ms"] > 0 and request["inference_duration_ms"] > 0
    stored = rows("SELECT model_version, predicted_cnt FROM predictions")
    assert stored == [{"model_version": "7", "predicted_cnt": 123.4}]


def test_batch_prediction_one_row_per_item(client):
    response = client.post("/predict/batch", json={"items": [VALID, {**VALID, "hr": 9}]})
    assert response.status_code == 200 and len(response.json()["predictions"]) == 2
    assert rows("SELECT item_count FROM api_requests")[0]["item_count"] == 2
    assert len(rows("SELECT * FROM predictions")) == 2


@pytest.mark.parametrize("payload", [
    {**VALID, "hr": 99},                 # hors bornes
    {**VALID, "temp": "chaud"},          # mauvais type
    {**VALID, "casual": 10},             # colonne interdite (fuite de cible)
    {k: v for k, v in VALID.items() if k != "hum"},   # champ manquant
])
def test_invalid_input_is_422_and_traced(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 and response.json()["error"] == "validation_error"
    request = rows("SELECT http_status, error_code FROM api_requests")[0]
    assert request == {"http_status": 422, "error_code": "validation_error"}
    assert rows("SELECT * FROM predictions") == []


def test_batch_size_limit(client):
    assert client.post("/predict/batch", json={"items": []}).status_code == 422
    assert client.post("/predict/batch", json={"items": [VALID] * 101}).status_code == 422


def test_model_unavailable_is_503(client):
    main.State.store.available = False
    response = client.post("/predict", json=VALID)
    assert response.status_code == 503 and response.json()["error"] == "model_unavailable"
    assert rows("SELECT error_code FROM api_requests")[0]["error_code"] == "model_unavailable"
    assert client.get("/health").status_code == 503


def test_database_down_does_not_block_predictions(client, monkeypatch):
    class Broken(Database):
        def executemany_tx(self, statements):
            raise RuntimeError("base coupée")
    monkeypatch.setattr(main.State, "database", Broken("sqlite:///nul"))
    assert client.post("/predict", json=VALID).status_code == 200
    assert main.State.trace_failures == 1


def test_feedback_lifecycle(client):
    prediction = client.post("/predict", json=VALID).json()
    payload = {"prediction_id": prediction["prediction_id"], "observed_cnt": 100}
    created = client.post("/feedback", json=payload)
    assert created.status_code == 201
    assert created.json()["absolute_error"] == pytest.approx(23.4) and created.json()["model_version"] == "7"
    assert client.post("/feedback", json=payload).status_code == 409          # un seul feedback
    assert client.post("/feedback", json={"prediction_id": "inconnue", "observed_cnt": 1}).status_code == 404
    assert client.post("/feedback", json={**payload, "observed_cnt": -5}).status_code == 422


def test_online_mae_per_version(client):
    for observed in (100, 150):
        prediction = client.post("/predict", json=VALID).json()
        client.post("/feedback", json={"prediction_id": prediction["prediction_id"], "observed_cnt": observed})
    mae = rows("SELECT p.model_version AS v, AVG(ABS(p.predicted_cnt - f.observed_cnt)) AS mae "
               "FROM feedbacks f JOIN predictions p USING (prediction_id) GROUP BY p.model_version")
    assert mae[0]["v"] == "7" and mae[0]["mae"] == pytest.approx((23.4 + 26.6) / 2)


def test_version_follows_the_store(client):
    assert client.get("/version").json()["model_version"] == "7"
    main.State.store.loaded = LoadedModel("bikeshare", "8", "run-def", FakeModel(), 0.0)   # alias déplacé
    assert client.get("/version").json()["model_version"] == "8"
    assert client.post("/predict", json=VALID).json()["model_version"] == "8"
