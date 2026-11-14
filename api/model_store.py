"""Modèle servi : chargé via l'alias MLflow `champion`, rechargé à chaud quand l'alias change.

L'identité exacte (version, run) est conservée avec le modèle : la réponse de l'API et les traces
indiquent toujours la version réellement utilisée, jamais seulement l'alias (qui est mutable).
"""
import logging
import threading
import time
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger("bikeshare.api")


class ModelUnavailable(RuntimeError):
    """Aucun modèle chargeable (MLflow injoignable, ou pas encore de champion)."""


@dataclass(frozen=True)
class LoadedModel:
    name: str
    version: str
    run_id: str
    model: object
    loaded_at: float

    def predict(self, frame: pd.DataFrame):
        return self.model.predict(frame)


class ModelStore:
    def __init__(self, name, alias, features, reload_interval):
        self.name, self.alias, self.features = name, alias, features
        self.reload_interval = reload_interval
        self._loaded = None
        self._last_check = 0.0
        self._lock = threading.Lock()

    def _load(self):
        import mlflow
        from bikeshare import tracking
        client = tracking.client()
        try:
            version = client.get_model_version_by_alias(self.name, self.alias)
        except Exception as exc:
            raise ModelUnavailable(f"alias {self.name}@{self.alias} introuvable : {exc}") from exc
        if self._loaded and self._loaded.version == str(version.version):
            return self._loaded
        try:
            model = mlflow.sklearn.load_model(f"models:/{self.name}/{version.version}")
        except Exception as exc:
            raise ModelUnavailable(f"chargement de {self.name} v{version.version} impossible : {exc}") from exc
        log.info("modèle chargé : %s v%s", self.name, version.version)
        return LoadedModel(self.name, str(version.version), version.run_id, model, time.time())

    def current(self):
        """Modèle courant ; relit l'alias au plus toutes les `reload_interval` secondes.

        Si la relecture échoue alors qu'un modèle est déjà chargé, on continue de le servir
        (dégradation maîtrisée) ; sans modèle du tout, l'erreur remonte (503).
        """
        now = time.monotonic()
        if self._loaded is None or now - self._last_check >= self.reload_interval:
            if self._lock.acquire(blocking=self._loaded is None):
                try:
                    self._last_check = now
                    try:
                        self._loaded = self._load()
                    except ModelUnavailable:
                        if self._loaded is None:
                            raise
                        log.warning("relecture de l'alias impossible : modèle %s conservé", self._loaded.version)
                finally:
                    self._lock.release()
        if self._loaded is None:
            raise ModelUnavailable("aucun modèle chargé")
        return self._loaded

    def frame(self, items):
        return pd.DataFrame([i.model_dump() for i in items], columns=self.features)
