"""Suivi MLflow : un run par exécution du pipeline, écriture explicite par `run_id`.

Le run est ouvert avant le travail et fermé en FAILED (étape + cause) sur toute exception : un
échec reste donc visible et rattaché à son lot. Le serveur choisit l'emplacement des artefacts.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from bikeshare.config import ROOT

MAX_VALUE = 500


class TrackingUnavailable(RuntimeError):
    """Serveur MLflow absent ou injoignable : le pipeline s'arrête avant toute publication."""


def _mlflow():
    import mlflow
    return mlflow


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    """Fixer l'URI depuis l'environnement ; refuser d'écrire dans un dossier `mlruns` local."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise TrackingUnavailable("MLFLOW_TRACKING_URI non défini : refus d'écrire dans un mlruns local.")
    for stream in (sys.stdout, sys.stderr):  # MLflow écrit des emojis : console Windows cp1252 sinon
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    mlflow = _mlflow()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI") or uri)
    return uri


def client():
    configure()
    return _mlflow().MlflowClient()


def _git(*args):
    try:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=20)
        return done.stdout.strip() if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_state():
    status = _git("status", "--porcelain", "--", "src", "flows", "api", "config", "pyproject.toml", "uv.lock")
    return {"git_commit": _git("rev-parse", "HEAD") or "unknown",
            "git_dirty": "unknown" if status is None else str(bool(status)).lower()}


def open_run(config, batch_id, extra_tags=None):
    """Créer le run du pipeline pour un lot ; retourne son identifiant."""
    uri = configure()
    mlflow_client = client()
    try:
        name = config["tracking"]["experiment"]
        found = mlflow_client.get_experiment_by_name(name)
        experiment_id = found.experiment_id if found else mlflow_client.create_experiment(name)
        now = datetime.now(timezone.utc)
        tags = {
            "batch_id": batch_id, "started_utc": now.isoformat(), "python": sys.version.split()[0],
            "platform": platform.platform(),
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
            "config_sha256": sha256_text(json.dumps(config, sort_keys=True, default=str)),
            **code_state(), **(extra_tags or {}),
        }
        run = mlflow_client.create_run(experiment_id, tags=tags,
                                       run_name=f"{batch_id}-{now:%Y%m%dT%H%M%S}")
        return run.info.run_id
    except Exception as exc:
        raise TrackingUnavailable(f"Serveur MLflow indisponible ({uri}) : {exc}") from exc


def _clip(value):
    return str(value)[:MAX_VALUE]


def log_params(run_id, params):
    from mlflow.entities import Param
    client().log_batch(run_id, params=[Param(k, _clip(v)) for k, v in params.items() if v is not None])


def log_metrics(run_id, metrics):
    from mlflow.entities import Metric
    now = int(time.time() * 1000)
    client().log_batch(run_id, metrics=[
        Metric(k, float(v), now, 0) for k, v in metrics.items()
        if isinstance(v, (int, float)) and v == v and abs(v) != float("inf")])


def set_tags(run_id, tags):
    from mlflow.entities import RunTag
    client().log_batch(run_id, tags=[RunTag(k, _clip(v)) for k, v in tags.items() if v is not None])


def log_json(run_id, payload, artifact_file):
    client().log_dict(run_id, payload, artifact_file)


def log_dataframe(run_id, frame, artifact_path, filename):
    """Archiver un snapshot CSV compressé ; retourne son SHA-256."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        frame.to_csv(path, index=False, compression="gzip")
        client().log_artifact(run_id, str(path), artifact_path)
        return sha256_file(path)


def log_dataset(run_id, frame, name, context, source, targets=None):
    """Associer un jeu au run (`log_input`) ; retourne le digest MLflow."""
    from mlflow.data import from_pandas
    from mlflow.entities import DatasetInput, InputTag
    dataset = from_pandas(frame.reset_index(drop=True), source=str(source), name=name, targets=targets)
    client().log_inputs(run_id, datasets=[DatasetInput(
        dataset=dataset._to_mlflow_entity(), tags=[InputTag("mlflow.data.context", context)])])
    return dataset.digest


def log_model(run_id, pipeline, example):
    """Journaliser le pipeline avec signature et exemple d'entrée ; le code `bikeshare` est archivé
    avec le modèle pour que le rechargement fonctionne dans un processus neuf."""
    mlflow = _mlflow()
    from mlflow.models import infer_signature
    configure()
    with mlflow.start_run(run_id=run_id):
        info = mlflow.sklearn.log_model(
            pipeline, name="model", serialization_format="cloudpickle",
            signature=infer_signature(example, pipeline.predict(example)),
            input_example=example, code_paths=[str(ROOT / "src" / "bikeshare")])
    client().update_run(run_id, status="RUNNING")  # le `with` a fermé le run : le pipeline continue
    return info.model_uri


def fail_run(run_id, step, exc):
    """Fermer le run en FAILED avec l'étape et la cause (jamais d'exception ici)."""
    try:
        set_tags(run_id, {"failed_step": step, "error_type": type(exc).__name__,
                          "error": str(exc)[:MAX_VALUE]})
        client().set_terminated(run_id, "FAILED")
    except Exception:
        pass


def finish_run(run_id):
    client().set_terminated(run_id, "FINISHED")


def last_completed_batch(experiment):
    """Plus grand numéro de lot dont le pipeline est allé au bout (run FINISHED), ou None."""
    mlflow_client = client()
    found = mlflow_client.get_experiment_by_name(experiment)
    if found is None:
        return None
    runs = mlflow_client.search_runs([found.experiment_id],
                                     filter_string="attributes.status = 'FINISHED'", max_results=1000)
    numbers = [int(r.data.tags["batch_id"].split("_")[1]) for r in runs if "batch_id" in r.data.tags]
    return max(numbers) if numbers else None
