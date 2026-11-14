"""API de prédiction Bikeshare.

    PYTHONPATH=src uv run uvicorn api.main:app --port 8000 [--workers 2]

Endpoints : POST /predict, POST /predict/batch, POST /feedback, GET /health, GET /version.
Le modèle vient de `models:/bikeshare@champion` ; chaque réponse identifie la version utilisée.
Chaque requête métier laisse une trace (table `api_requests`, y compris les 4xx/5xx), chaque
prédiction une ligne `predictions` liée à sa version.

Choix de fiabilité : si la base est indisponible, les PRÉDICTIONS sont quand même servies (la trace
manquante est comptée et visible dans /health) ; un FEEDBACK, lui, est refusé (503) car sa seule
raison d'être est d'être conservé.
"""
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from fastapi import FastAPI, Header, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from api.model_store import ModelStore, ModelUnavailable  # noqa: E402
from api.schemas import (BatchRequest, BatchResponse, BikeFeatures, FeedbackRequest, FeedbackResponse,  # noqa: E402
                         PredictResponse, Prediction)
from bikeshare import db as dbmod  # noqa: E402
from bikeshare.config import load_config  # noqa: E402

log = logging.getLogger("bikeshare.api")
CONFIG = load_config()
ENDPOINTS = {"/predict", "/predict/batch", "/feedback"}


class ApiError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


class State:
    store = ModelStore(
        name=os.getenv("MODEL_NAME", CONFIG["model"]["registered_name"]),
        alias=os.getenv("MODEL_ALIAS", CONFIG["model"]["champion_alias"]),
        features=CONFIG["features"]["raw"],
        reload_interval=float(os.getenv("RELOAD_INTERVAL", CONFIG["api"]["reload_interval_seconds"])))
    database = dbmod.from_env()
    trace_failures = 0
    revision = os.getenv("API_REVISION", "unknown")
    instance = uuid.uuid4().hex[:8]


@asynccontextmanager
async def lifespan(app):
    if State.database is not None:
        try:
            State.database.init_schema()
        except Exception as exc:
            log.warning("schéma non initialisé : %s", exc)
    try:
        State.store.current()      # charge le champion au démarrage ; sinon on réessaie à la requête
    except ModelUnavailable as exc:
        log.warning("démarrage sans modèle : %s", exc)
    yield


app = FastAPI(title="Bikeshare API", version="1.0", lifespan=lifespan)


class Trace:
    """Contexte d'une requête métier : horloge monotone, lignes à persister."""

    def __init__(self, endpoint, campaign):
        self.request_id = uuid.uuid4().hex
        self.endpoint, self.campaign = endpoint, campaign or "manual"
        self.t0 = time.perf_counter()
        self.started_at = datetime.now(timezone.utc)
        self.item_count = None
        self.inference_ms = None
        self.model_version = None
        self.predictions = []

    def request_statement(self, status, error_code):
        duration = (time.perf_counter() - self.t0) * 1000
        return ("INSERT INTO api_requests (request_id, started_at, endpoint, http_status, item_count, "
                "processing_duration_ms, inference_duration_ms, model_version, error_code, campaign_id, "
                "worker_pid) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (self.request_id, self.started_at, self.endpoint, status, self.item_count, duration,
                 self.inference_ms, self.model_version, error_code, self.campaign, os.getpid()))


def persist(trace, status, error_code=None, extra=()):
    """Écrire la trace de la requête (+ prédictions) en une transaction. Jamais bloquant pour /predict."""
    if State.database is None:
        return
    statements = [trace.request_statement(status, error_code)]
    now = datetime.now(timezone.utc)
    for index, row in enumerate(trace.predictions):
        statements.append((
            "INSERT INTO predictions (prediction_id, request_id, item_index, model_name, model_version, "
            "input_features, predicted_cnt, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (row["prediction_id"], trace.request_id, index, row["model_name"], row["model_version"],
             row["input"], row["predicted_cnt"], now)))
    statements.extend(extra)
    State.database.executemany_tx(statements)


def persist_best_effort(trace, status, error_code=None):
    try:
        persist(trace, status, error_code)
    except Exception as exc:
        State.trace_failures += 1
        log.warning("trace non écrite (%s) : %s", trace.request_id, exc)


def run_predictions(trace, items):
    loaded = State.store.current()
    frame = State.store.frame(items)
    t = time.perf_counter()
    values = loaded.predict(frame)
    trace.inference_ms = (time.perf_counter() - t) * 1000
    trace.item_count, trace.model_version = len(items), loaded.version
    results = []
    for item, value in zip(items, values):
        row = {"prediction_id": uuid.uuid4().hex, "model_name": loaded.name, "model_version": loaded.version,
               "predicted_cnt": max(0.0, float(value)), "input": item.model_dump_json()}
        trace.predictions.append(row)
        results.append(Prediction(prediction_id=row["prediction_id"], predicted_cnt=row["predicted_cnt"]))
    return loaded, results


@app.post("/predict", response_model=PredictResponse)
def predict(item: BikeFeatures, x_campaign_id: str | None = Header(default=None)):
    trace = Trace("/predict", x_campaign_id)
    try:
        loaded, (result,) = run_predictions(trace, [item])
    except ModelUnavailable as exc:
        persist_best_effort(trace, 503, "model_unavailable")
        raise ApiError(503, "model_unavailable", str(exc)) from exc
    persist_best_effort(trace, 200)
    return PredictResponse(request_id=trace.request_id, model_name=loaded.name,
                           model_version=loaded.version, **result.model_dump())


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest, x_campaign_id: str | None = Header(default=None)):
    trace = Trace("/predict/batch", x_campaign_id)
    try:
        loaded, results = run_predictions(trace, body.items)
    except ModelUnavailable as exc:
        persist_best_effort(trace, 503, "model_unavailable")
        raise ApiError(503, "model_unavailable", str(exc)) from exc
    persist_best_effort(trace, 200)
    return BatchResponse(request_id=trace.request_id, model_name=loaded.name,
                         model_version=loaded.version, predictions=results)


@app.post("/feedback", response_model=FeedbackResponse, status_code=201)
def feedback(body: FeedbackRequest, x_campaign_id: str | None = Header(default=None)):
    trace = Trace("/feedback", x_campaign_id)
    database = State.database
    if database is None:
        raise ApiError(503, "persistence_disabled", "aucune base configurée : feedback impossible")
    try:
        rows = database.query("SELECT model_version, predicted_cnt FROM predictions WHERE prediction_id = %s",
                              (body.prediction_id,))
        if not rows:
            persist_best_effort(trace, 404, "unknown_prediction")
            raise ApiError(404, "unknown_prediction", f"prédiction inconnue : {body.prediction_id}")
        if database.query("SELECT 1 AS x FROM feedbacks WHERE prediction_id = %s", (body.prediction_id,)):
            persist_best_effort(trace, 409, "feedback_exists")
            raise ApiError(409, "feedback_exists", "un feedback existe déjà pour cette prédiction")
        trace.item_count, trace.model_version = 1, rows[0]["model_version"]
        feedback_id = uuid.uuid4().hex
        persist(trace, 201, extra=[(
            "INSERT INTO feedbacks (feedback_id, prediction_id, request_id, observed_cnt, created_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (feedback_id, body.prediction_id, trace.request_id, body.observed_cnt, datetime.now(timezone.utc)))])
    except dbmod.DatabaseUnavailable as exc:
        raise ApiError(503, "database_unavailable", str(exc)) from exc
    return FeedbackResponse(feedback_id=feedback_id, prediction_id=body.prediction_id,
                            model_version=rows[0]["model_version"],
                            absolute_error=abs(rows[0]["predicted_cnt"] - body.observed_cnt))


@app.get("/health")
def health():
    try:
        loaded = State.store.current()
        model_ok = True
    except ModelUnavailable:
        loaded, model_ok = None, False
    database = None
    if State.database is not None:
        try:
            State.database.query("SELECT 1 AS x")
            database = True
        except Exception:
            database = False
    status = "ok" if model_ok and database is not False else ("degraded" if model_ok else "unavailable")
    return JSONResponse({"status": status, "model_loaded": model_ok,
                         "model_version": loaded.version if loaded else None, "database": database,
                         "trace_failures": State.trace_failures}, status_code=200 if model_ok else 503)


@app.get("/version")
def version():
    try:
        loaded = State.store.current()
    except ModelUnavailable as exc:
        raise ApiError(503, "model_unavailable", str(exc)) from exc
    return {"model_name": loaded.name, "model_version": loaded.version, "alias": State.store.alias,
            "mlflow_run_id": loaded.run_id, "api_revision": State.revision, "instance": State.instance}


def error_response(status, code, message, request_id=None):
    return JSONResponse(status_code=status, content={"error": code, "message": message, "request_id": request_id})


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return error_response(exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """422 clair + trace de la requête rejetée (le corps n'a pas été accepté : aucune prédiction)."""
    trace = Trace(request.url.path, request.headers.get("x-campaign-id"))
    if request.url.path in ENDPOINTS:
        persist_best_effort(trace, 422, "validation_error")
    errors = [{"champ": ".".join(str(p) for p in e["loc"][1:]), "probleme": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_error",
                                                  "message": "entrée invalide", "details": errors,
                                                  "request_id": trace.request_id})
