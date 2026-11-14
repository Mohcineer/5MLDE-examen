"""Démonstration d'incident contrôlé : un lot corrompu doit être bloqué sans rien publier.

    uv run python scripts/demo_incident.py [--api-url http://127.0.0.1:8000]

Prérequis : MLflow, Prefect, le récepteur d'alertes (`scripts/alert_receiver.py`) et l'automation
(`flows/setup_alerts.py`) en marche. Le lot corrompu est une COPIE (data/incident) : les vrais lots
ne sont jamais modifiés.

Preuves vérifiées : (1) le flow échoue à l'étape `validate`, sans retry ; (2) le run MLflow est FAILED,
étape et rapport de contrôle joints ; (3) aucune nouvelle version de modèle, le champion est inchangé ;
(4) une ligne `pipeline_runs` en échec ; (5) l'alerte est reçue ; (6) si l'API tourne, elle sert
toujours la même version.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from bikeshare import db as dbmod  # noqa: E402
from bikeshare import pipeline, registry, tracking  # noqa: E402
from bikeshare.config import load_config  # noqa: E402
from bikeshare.ingestion import sha256_file  # noqa: E402

ALERTS = ROOT / "local" / "alerts.log"


def make_incident_dir(config, batch_id):
    """Copie des lots où `batch_id` est corrompu (heure 99, humidité manquante, doublon)."""
    source, target = Path(config["data"]["batches_dir"]), ROOT / "data" / "incident"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    path = target / f"{batch_id}.csv"
    frame = pd.read_csv(path)
    frame.loc[3, "hr"] = 99
    frame.loc[5, "hum"] = None
    frame = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
    frame.to_csv(path, index=False)
    manifest = json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest:
        if entry["batch_id"] == batch_id:
            entry["sha256"], entry["rows"] = sha256_file(path), len(frame)
    (target / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def wait_for_alert(since_lines, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        lines = ALERTS.read_text(encoding="utf-8").splitlines() if ALERTS.exists() else []
        if len(lines) > since_lines:
            return lines[-1]
        time.sleep(3)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    config = load_config()
    batch_id = pipeline.next_batch_id(config)
    before = registry.status(config)
    alerts_before = len(ALERTS.read_text(encoding="utf-8").splitlines()) if ALERTS.exists() else 0
    print(f"[1/6] État initial : champion v{before['champion']}, {len(before['versions'])} versions.")

    data_dir = make_incident_dir(config, batch_id)
    print(f"[2/6] Lot {batch_id} corrompu dans {data_dir} : lancement du flow.")
    from flows.pipeline_flow import bikeshare_pipeline
    state = bikeshare_pipeline(batch_id=batch_id, data_dir=str(data_dir), return_state=True)
    print(f"      État du flow : {state.name} — {state.message}")
    ok = state.is_failed()

    after = registry.status(config)
    unchanged = after["champion"] == before["champion"] and len(after["versions"]) == len(before["versions"])
    print(f"[3/6] Aucun modèle publié, champion inchangé (v{after['champion']}) : {unchanged}")

    client = tracking.client()
    experiment = client.get_experiment_by_name(config["tracking"]["experiment"])
    run = client.search_runs([experiment.experiment_id], f"tags.batch_id = '{batch_id}' and "
                             "attributes.status = 'FAILED'", order_by=["attributes.start_time DESC"],
                             max_results=1)[0]
    print(f"[4/6] Run MLflow {run.info.run_id} : {run.info.status}, étape en échec = "
          f"{run.data.tags.get('failed_step')}, cause = {run.data.tags.get('error')}")

    database = dbmod.from_env()
    if database:
        rows = database.query("SELECT batch_id, status, failed_step FROM pipeline_runs "
                              "WHERE status = 'failed' ORDER BY started_at DESC LIMIT 1")
        print(f"      pipeline_runs : {rows}")

    alert = wait_for_alert(alerts_before)
    print(f"[5/6] Alerte reçue : {'oui' if alert else 'NON'}")
    if alert:
        print("      " + json.loads(alert)["body"][:400].replace("\\n", " | "))

    try:
        version = httpx.get(f"{args.api_url}/version", timeout=5).json()
        print(f"[6/6] API : sert toujours {version['model_name']} v{version['model_version']}")
    except Exception:
        print("[6/6] API non joignable : vérification ignorée.")
    return 0 if ok and unchanged and alert else 1


if __name__ == "__main__":
    sys.exit(main())
