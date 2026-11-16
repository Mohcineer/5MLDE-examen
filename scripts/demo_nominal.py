"""Démonstration du fonctionnement nominal, de bout en bout.

    uv run python scripts/demo_nominal.py [--api-url http://127.0.0.1:8000] [--predictions 20] [--rollback]

Enchaîne : lot valide -> flow Prefect (validation, entraînement, évaluation) -> qualification ->
promotion -> l'API recharge le champion -> prédictions -> feedbacks (valeurs réelles du lot suivant)
-> requêtes SQL (traces, versions, durées, MAE en ligne). Avec --rollback : retour arrière, puis
constat que l'API sert à nouveau la version précédente.
Prérequis : MLflow, Prefect, API démarrés (`scripts/local_stack.py start` ou `docker compose up`).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import httpx  # noqa: E402

from bikeshare import db as dbmod  # noqa: E402
from bikeshare import pipeline, registry  # noqa: E402
from bikeshare.config import load_config  # noqa: E402
from bikeshare.ingestion import load_batch  # noqa: E402


FLOATS = {"temp", "atemp", "hum", "windspeed"}


def wait_for_version(api, version, timeout=60):
    """L'API relit l'alias `champion` périodiquement : attendre qu'elle serve `version`."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        served = httpx.get(f"{api}/version", timeout=10).json()["model_version"]
        if str(served) == str(version):
            return True
        time.sleep(2)
    return False


def table(rows, title):
    print(f"   {title}")
    for row in rows:
        print("     " + " | ".join(f"{k}={v}" for k, v in row.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--predictions", type=int, default=20)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    config = load_config()
    api = args.api_url
    feature_cols = config["features"]["raw"]

    before = registry.status(config)
    batch_id = pipeline.next_batch_id(config)
    print(f"[1/7] Avant : champion v{before['champion']} ; prochain lot à traiter : {batch_id}")

    from flows.pipeline_flow import bikeshare_pipeline
    result = bikeshare_pipeline(batch_id=batch_id)
    print(f"[2/7] Flow terminé : version v{result['model_version']} "
          f"({result['qualification']}), promue = {bool(result['promoted'])}, run MLflow {result['mlflow_run_id']}")

    after = registry.status(config)
    print(f"[3/7] Registre : champion v{after['champion']} ; dernier événement : {after['log'][-1]}")

    if wait_for_version(api, after["champion"]):
        print(f"[4/7] L'API sert maintenant la version v{after['champion']} (rechargement à chaud).")
    else:
        print("[4/7] L'API n'a pas rechargé le champion à temps.")
        return 1

    next_id = f"batch_{int(batch_id.split('_')[1]) + 1:04d}"
    future = load_batch(config, next_id).head(args.predictions)   # semaine suivante : valeurs réelles connues
    served, feedbacks = set(), 0
    campaign = {"X-Campaign-Id": "demo-nominal"}
    for _, row in future.iterrows():
        payload = {col: float(row[col]) if col in FLOATS else int(row[col]) for col in feature_cols}
        reply = httpx.post(f"{api}/predict", json=payload, headers=campaign, timeout=30).json()
        served.add(reply["model_version"])
        sent = httpx.post(f"{api}/feedback", headers=campaign, timeout=30,
                          json={"prediction_id": reply["prediction_id"], "observed_cnt": float(row["cnt"])})
        feedbacks += sent.status_code == 201
    print(f"[5/7] {len(future)} prédictions (versions servies : {sorted(served)}), {feedbacks} feedbacks "
          f"(valeurs réelles de {next_id}).")

    database = dbmod.from_env()
    if database:
        print("[6/7] Traces en base :")
        table(database.query("SELECT model_version, count(*) AS predictions FROM predictions GROUP BY 1 ORDER BY 1"),
              "prédictions par version")
        table(database.query(
            "SELECT p.model_version AS version, count(*) AS feedbacks, "
            "round(CAST(avg(abs(p.predicted_cnt - f.observed_cnt)) AS NUMERIC), 2) AS mae_en_ligne "
            "FROM feedbacks f JOIN predictions p ON p.prediction_id = f.prediction_id GROUP BY 1 ORDER BY 1"),
            "MAE en ligne par version")
        table(database.query(
            "SELECT endpoint, http_status, count(*) AS n, round(CAST(avg(processing_duration_ms) AS NUMERIC), 1) AS ms_moyen "
            "FROM api_requests WHERE campaign_id = 'demo-nominal' GROUP BY 1, 2 ORDER BY 1, 2"), "requêtes de la démo")
        table(database.query("SELECT batch_id, status, round(CAST(duration_s AS NUMERIC), 1) AS duree_s, model_version, "
                             "qualification, promoted FROM pipeline_runs ORDER BY started_at DESC LIMIT 3"),
              "dernières exécutions du pipeline")
    else:
        print("[6/7] DATABASE_URL absent : traces non vérifiées.")

    if args.rollback:
        undone = registry.rollback(config)
        ok = wait_for_version(api, undone["version"])
        print(f"[7/7] Retour arrière : champion v{undone['previous']} -> v{undone['version']} ; "
              f"l'API suit : {ok}")
    else:
        print("[7/7] (retour arrière non demandé : option --rollback)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
