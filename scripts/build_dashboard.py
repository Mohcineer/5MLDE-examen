"""Générer le dashboard Grafana provisionné (grafana/dashboards/bikeshare.json).

    uv run python scripts/build_dashboard.py

Les requêtes visent PostgreSQL (datasource `bikeshare-postgres`). Elles sont écrites ici pour rester
relisibles ; le JSON produit est versionné dans Git et monté dans le conteneur Grafana.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "grafana" / "dashboards" / "bikeshare.json"
DS = {"type": "postgres", "uid": "bikeshare-postgres"}
_ids = iter(range(1, 100))


def target(sql, fmt="time_series"):
    return {"refId": "A", "datasource": DS, "editorMode": "code", "format": fmt, "rawQuery": True, "rawSql": sql}


def panel(title, kind, sql, x, y, w, h, fmt="time_series", unit=None, desc="", options=None, overrides=None):
    defaults = {"unit": unit} if unit else {}
    return {"id": next(_ids), "title": title, "type": kind, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [target(sql, fmt)],
            "fieldConfig": {"defaults": defaults, "overrides": overrides or []}, "options": options or {}}


def row(title, y):
    return {"id": next(_ids), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


PANELS = [
    row("Traitements (pipeline_runs)", 0),
    panel("Exécutions : succès / échecs", "stat",
          "SELECT status AS metric, count(*) AS value FROM pipeline_runs "
          "WHERE $__timeFilter(started_at) AND status <> 'running' GROUP BY status",
          0, 1, 6, 5, fmt="table", desc="Nombre d'exécutions du pipeline sur la période.",
          options={"reduceOptions": {"calcs": ["lastNotNull"], "values": True, "fields": "/^value$/",
                                     "limit": 10}, "textMode": "value_and_name"}),
    panel("Durée des traitements", "timeseries",
          "SELECT started_at AS time, duration_s AS \"durée (s)\" FROM pipeline_runs "
          "WHERE $__timeFilter(started_at) AND status = 'success' ORDER BY 1",
          6, 1, 9, 5, unit="s"),
    panel("Échecs par étape", "barchart",
          "SELECT coalesce(failed_step, 'inconnue') AS étape, count(*) AS échecs FROM pipeline_runs "
          "WHERE $__timeFilter(started_at) AND status = 'failed' GROUP BY 1 ORDER BY 2 DESC",
          15, 1, 9, 5, fmt="table", desc="Étape du pipeline où l'exécution s'est arrêtée."),
    panel("Dernières exécutions", "table",
          "SELECT started_at AS début, batch_id AS lot, status AS état, round(duration_s::numeric, 1) AS \"durée (s)\", "
          "failed_step AS \"étape en échec\", left(error, 80) AS motif, model_version AS version, "
          "qualification, promoted AS promue, flow_run_id, mlflow_run_id FROM pipeline_runs "
          "WHERE $__timeFilter(started_at) ORDER BY started_at DESC LIMIT 20",
          0, 6, 24, 8, fmt="table"),

    row("Utilisation du service (api_requests)", 14),
    panel("Requêtes par minute et par endpoint", "timeseries",
          "SELECT $__timeGroupAlias(started_at, 1m), endpoint AS metric, count(*) AS value FROM api_requests "
          "WHERE $__timeFilter(started_at) GROUP BY 1, 2 ORDER BY 1", 0, 15, 12, 8),
    panel("Latence de traitement p50 / p95 / p99", "timeseries",
          "SELECT $__timeGroupAlias(started_at, 1m), "
          "percentile_cont(0.5) WITHIN GROUP (ORDER BY processing_duration_ms) AS p50, "
          "percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_duration_ms) AS p95, "
          "percentile_cont(0.99) WITHIN GROUP (ORDER BY processing_duration_ms) AS p99 "
          "FROM api_requests WHERE $__timeFilter(started_at) AND http_status < 400 GROUP BY 1 ORDER BY 1",
          12, 15, 12, 8, unit="ms",
          desc="Du début du traitement HTTP à la fin du calcul ; exclut le transport réseau et l'écriture de la trace."),
    panel("Taux d'erreur", "timeseries",
          "SELECT $__timeGroupAlias(started_at, 1m), avg((http_status >= 400)::int)::float AS \"taux d'erreur\" "
          "FROM api_requests WHERE $__timeFilter(started_at) GROUP BY 1 ORDER BY 1",
          0, 23, 8, 7, unit="percentunit"),
    panel("Erreurs par code", "table",
          "SELECT http_status AS statut, error_code AS code, endpoint, count(*) AS nombre FROM api_requests "
          "WHERE $__timeFilter(started_at) AND http_status >= 400 GROUP BY 1, 2, 3 ORDER BY 4 DESC",
          8, 23, 8, 7, fmt="table"),
    panel("Prédictions par version du modèle", "piechart",
          "SELECT 'v' || model_version AS version, count(*) AS prédictions FROM predictions "
          "WHERE $__timeFilter(created_at) GROUP BY model_version ORDER BY model_version",
          16, 23, 8, 7, fmt="table",
          options={"reduceOptions": {"calcs": ["lastNotNull"], "values": True}, "legend": {"displayMode": "table"}}),

    row("Qualité en ligne (feedbacks)", 30),
    panel("MAE en ligne par version", "table",
          "SELECT 'v' || p.model_version AS version, count(*) AS feedbacks, "
          "round(avg(abs(p.predicted_cnt - f.observed_cnt))::numeric, 2) AS \"MAE en ligne\", "
          "round(sqrt(avg(power(p.predicted_cnt - f.observed_cnt, 2)))::numeric, 2) AS \"RMSE en ligne\" "
          "FROM feedbacks f JOIN predictions p USING (prediction_id) "
          "WHERE $__timeFilter(f.created_at) GROUP BY p.model_version ORDER BY p.model_version",
          0, 31, 12, 7, fmt="table",
          desc="Erreur entre la prédiction servie et la valeur réelle déclarée par feedback."),
    panel("Feedbacks reçus par minute", "timeseries",
          "SELECT $__timeGroupAlias(created_at, 1m), count(*) AS feedbacks FROM feedbacks "
          "WHERE $__timeFilter(created_at) GROUP BY 1 ORDER BY 1", 12, 31, 12, 7),
]

DASHBOARD = {
    "uid": "bikeshare-ops", "title": "Bikeshare : traitements et service", "tags": ["bikeshare", "mlops"],
    "timezone": "browser", "schemaVersion": 39, "version": 1, "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"}, "editable": False, "panels": PANELS,
    "annotations": {"list": []}, "templating": {"list": []},
    "description": "Traitements du pipeline, utilisation, performances et erreurs de l'API de prévision.",
}

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(DASHBOARD, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(PANELS)} éléments écrits dans {OUT}")
