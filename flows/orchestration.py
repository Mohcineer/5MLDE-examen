"""Aides Prefect : politique de retry (transitoire uniquement) et références dans les tags du flow run."""
import httpx
from prefect.runtime import flow_run

from bikeshare.db import DatabaseUnavailable
from bikeshare.ingestion import IngestionError
from bikeshare.tracking import TrackingUnavailable
from bikeshare.validation import DataQualityError

# Données invalides ou lot absent : réessayer ne changerait rien, on échoue tout de suite.
PERMANENT = (DataQualityError, IngestionError, ValueError, KeyError)
# Dépendance momentanément injoignable : un retry borné a un sens.
TRANSIENT = (ConnectionError, TimeoutError, httpx.TransportError, TrackingUnavailable, DatabaseUnavailable)
TRANSIENT_MARKERS = ("Connection", "timed out", "Max retries exceeded", "HTTPConnectionPool", "503", "502")


def is_transient(exc):
    """Vrai pour une indisponibilité passagère (MLflow, base, réseau) ; jamais pour une donnée invalide."""
    if isinstance(exc, PERMANENT):
        return False
    if isinstance(exc, TRANSIENT):
        return True
    return any(marker in str(exc) for marker in TRANSIENT_MARKERS)


def retry_on_transient(task, task_run, state):
    """`retry_condition_fn` Prefect : ne relance que les erreurs transitoires."""
    try:
        state.result()
    except Exception as exc:
        return is_transient(exc)
    return False


def annotate_flow_run(**refs):
    """Publier des références (batch, run MLflow, étape en erreur) dans les TAGS du flow run.

    Le message d'alerte d'une automation ne voit que les métadonnées du run, pas les variables du
    processus. Sans contexte Prefect ou si l'API est injoignable, l'annotation est ignorée : elle ne
    doit jamais faire échouer ni masquer l'échec du pipeline.
    """
    try:
        from prefect.client.orchestration import get_client
        run_id = flow_run.id
        if not run_id:
            return
        with get_client(sync_client=True) as client:
            current = client.read_flow_run(run_id).tags or []   # tags COURANTS du serveur
            kept = [t for t in current if t.split(":", 1)[0] not in refs]
            tags = kept + [f"{key}:{value}" for key, value in refs.items() if value is not None]
            client.update_flow_run(run_id, tags=tags)
    except Exception:
        return
