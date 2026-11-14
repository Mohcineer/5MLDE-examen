"""Étapes du pipeline : ingest -> validate -> preprocess -> train -> evaluate -> publish.

Chaque étape est une fonction ordinaire (testable sans Prefect). `flows/pipeline_flow.py` les enrobe
dans des tâches Prefect ; `run_pipeline` les enchaîne sans orchestrateur (tests, démos, secours).
`PipelineContext.step` mémorise l'étape en cours : c'est elle qui est rapportée en cas d'échec.
"""
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bikeshare import db as dbmod
from bikeshare import registry, tracking
from bikeshare.config import load_config, load_policy
from bikeshare.evaluation import cross_validate, evaluate
from bikeshare.ingestion import IngestionError, load_batch
from bikeshare.preprocessing import history_before, split_xy
from bikeshare.training import train
from bikeshare.validation import ensure_valid

STEPS = ("ingest", "validate", "preprocess", "train", "evaluate", "publish")


@dataclass
class PipelineContext:
    config: dict
    batch_id: str
    run_id: str = None
    step: str = None
    flow_run_id: str = None
    run_key: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.perf_counter)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: dict = field(default_factory=dict)


def batch_ids(config):
    manifest = json.loads((Path(config["data"]["batches_dir"]) / "MANIFEST.json")
                          .read_text(encoding="utf-8"))
    return [e["batch_id"] for e in manifest], [e for e in manifest if e["role"] == "reference"]


def next_batch_id(config):
    """Prochain lot à traiter : le suivant du dernier pipeline terminé (premier lot entrant au départ)."""
    ids, references = batch_ids(config)
    done = tracking.last_completed_batch(config["tracking"]["experiment"])
    number = (done + 1) if done else len(references) + 1
    candidate = f"batch_{number:04d}"
    if candidate not in ids:
        raise IngestionError("Tous les lots ont été traités : plus de nouvelles données à rejouer.")
    return candidate


# --- étapes --------------------------------------------------------------------------------------

def step_open(ctx, extra_tags=None):
    """Ouvrir le run MLflow (avant tout travail) et enregistrer l'exécution comme `running`."""
    ctx.step = "open_run"
    tags = {"prefect_flow_run_id": ctx.flow_run_id} if ctx.flow_run_id else {}
    ctx.run_id = tracking.open_run(ctx.config, ctx.batch_id, {**tags, **(extra_tags or {})})
    _record(ctx, "running")


def step_ingest(ctx):
    ctx.step = "ingest"
    config = ctx.config
    ids, references = batch_ids(config)
    if ctx.batch_id not in ids:
        raise IngestionError(f"Lot inconnu : {ctx.batch_id}")
    batch = load_batch(config, ctx.batch_id)
    previous = {i: load_batch(config, i) for i in ids if i < ctx.batch_id}
    reference = pd.concat([previous[r["batch_id"]] for r in references if r["batch_id"] in previous],
                          ignore_index=True) if previous else None
    return {"batch": batch, "previous": previous, "reference": reference}


def step_validate(ctx, data):
    """Contrôles bloquants : un lot non conforme lève DataQualityError et arrête tout le pipeline."""
    ctx.step = "validate"
    report = ensure_valid(data["batch"], ctx.config, ctx.batch_id, data["reference"])
    tracking.log_json(ctx.run_id, report.to_dict(), "data/validation_report.json")
    return report


def step_preprocess(ctx, data):
    ctx.step = "preprocess"
    history = history_before({**data["previous"], ctx.batch_id: data["batch"]}, ctx.batch_id)
    X_train, y_train = split_xy(history, ctx.config)
    X_val, y_val = split_xy(data["batch"], ctx.config)
    return {"history": history, "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}


def step_train(ctx, prep):
    ctx.step = "train"
    params = ctx.config["model"]["params"]
    model = train(prep["X_train"], prep["y_train"], params)
    cv = cross_validate(prep["X_train"], prep["y_train"], params, ctx.config["model"]["cv_splits"])
    return {"model": model, "cv": cv}


def step_evaluate(ctx, data, prep, trained):
    """Métriques (baseline incluse), traçabilité des données, modèle rechargé dans un processus neuf."""
    ctx.step = "evaluate"
    config, run_id = ctx.config, ctx.run_id
    policy = load_policy(config)
    metrics = evaluate(trained["model"], prep["X_train"], prep["y_train"], prep["X_val"], prep["y_val"],
                       trained["cv"])
    train_ids = sorted(data["previous"])
    manifest = json.loads((Path(config["data"]["batches_dir"]) / "MANIFEST.json").read_text("utf-8"))
    shas = {e["batch_id"]: e["sha256"] for e in manifest}
    tracking.log_params(run_id, {
        **{f"model.{k}": v for k, v in config["model"]["params"].items()},
        "protocol_id": policy["protocol_id"], "train_batches": f"{train_ids[0]}..{train_ids[-1]}",
        "train_rows": len(prep["X_train"]), "val_batch": ctx.batch_id, "val_rows": len(prep["X_val"]),
        "features": ",".join(config["features"]["raw"]), "cv_splits": config["model"]["cv_splits"]})
    tracking.log_metrics(run_id, metrics)
    tracking.set_tags(run_id, {
        "val_batch_sha256": shas[ctx.batch_id],
        "train_data_sha256": tracking.sha256_text("".join(shas[i] for i in train_ids)),
        "policy_id": policy["policy_id"]})
    tracking.log_dataset(run_id, prep["history"], "train", "training", config["data"]["batches_dir"],
                         targets=config["target"])
    tracking.log_dataset(run_id, data["batch"], ctx.batch_id, "validation", config["data"]["batches_dir"],
                         targets=config["target"])
    tracking.log_dataframe(run_id, data["batch"], "data", f"{ctx.batch_id}.csv.gz")
    example = prep["X_val"].head(5)
    model_uri = tracking.log_model(run_id, trained["model"], example)
    reload_ok = registry.reload_check(model_uri, example)
    evidence = {"batch_id": ctx.batch_id, "protocol_id": policy["protocol_id"], "metrics": metrics,
                "model_uri": model_uri, "reload_ok": reload_ok}
    tracking.log_json(run_id, evidence, registry.EVIDENCE_FILE)
    return {"metrics": metrics, "model_uri": model_uri, "reload_ok": reload_ok}


def step_publish(ctx, evaluation):
    """Enregistrer la version, la qualifier, puis tenter la mise en service si la config l'autorise."""
    ctx.step = "publish"
    config = ctx.config
    version = registry.register(config, ctx.run_id, evaluation["model_uri"], ctx.batch_id,
                                evaluation["reload_ok"])
    report = registry.qualify(config, version.version)
    result = {"model_version": version.version, "qualification": report["status"], "promoted": 0,
              "promotion_note": None}
    if config["pipeline"]["auto_promote"] and report["status"] == "production-ready":
        try:
            registry.promote(config, version.version, actor="pipeline")
            result["promoted"] = 1
        except registry.PromotionRefused as exc:
            result["promotion_note"] = str(exc)  # refus normal : l'ancien champion reste en place
    tracking.set_tags(ctx.run_id, {"model_version": version.version, "qualification": report["status"],
                                   "promoted": result["promoted"],
                                   "promotion_note": result["promotion_note"]})
    return result


# --- suivi des exécutions ---------------------------------------------------------------------

def _record(ctx, status, error=None):
    """Ligne `pipeline_runs` (source du dashboard). Best-effort : ne masque jamais l'issue du pipeline."""
    database = dbmod.from_env()
    if database is None:
        return
    now = datetime.now(timezone.utc)
    done = status != "running"
    try:
        dbmod.record_pipeline_run(database, {
            "run_key": ctx.flow_run_id or ctx.run_key, "batch_id": ctx.batch_id, "status": status,
            "started_at": ctx.started_at, "finished_at": now if done else None,
            "duration_s": time.perf_counter() - ctx.started if done else None,
            "failed_step": ctx.step if status == "failed" else None,
            "error": str(error)[:500] if error else None, "flow_run_id": ctx.flow_run_id,
            "mlflow_run_id": ctx.run_id, "model_version": ctx.result.get("model_version"),
            "qualification": ctx.result.get("qualification"), "promoted": ctx.result.get("promoted")})
    except Exception:
        pass


def step_close_ok(ctx, result):
    ctx.result = result
    tracking.finish_run(ctx.run_id)
    _record(ctx, "success")


def step_close_failed(ctx, exc):
    """Run MLflow en FAILED (étape + cause) et exécution marquée en échec."""
    if ctx.run_id:
        report = getattr(exc, "report", None)  # DataQualityError : le rapport accompagne le run FAILED
        if report is not None:
            try:
                tracking.log_json(ctx.run_id, report.to_dict(), "data/validation_report.json")
            except Exception:
                pass
        tracking.fail_run(ctx.run_id, ctx.step, exc)
    _record(ctx, "failed", exc)


def run_pipeline(config=None, batch_id=None, flow_run_id=None):
    """Enchaîner les étapes sans orchestrateur. Lève l'exception d'origine après avoir tracé l'échec."""
    config = config or load_config()
    ctx = PipelineContext(config=config, batch_id=batch_id or next_batch_id(config), flow_run_id=flow_run_id)
    try:
        step_open(ctx)
        data = step_ingest(ctx)
        step_validate(ctx, data)
        prep = step_preprocess(ctx, data)
        trained = step_train(ctx, prep)
        evaluation = step_evaluate(ctx, data, prep, trained)
        published = step_publish(ctx, evaluation)
        step_close_ok(ctx, {**published, "metrics": evaluation["metrics"]})
    except Exception as exc:
        step_close_failed(ctx, exc)
        raise
    return {"batch_id": ctx.batch_id, "mlflow_run_id": ctx.run_id, **ctx.result}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Traiter un lot sans orchestrateur")
    parser.add_argument("batch_id", nargs="?")
    args = parser.parse_args(argv)
    out = run_pipeline(batch_id=args.batch_id)
    out.pop("metrics", None)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
