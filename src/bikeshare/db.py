"""Accès base de données portable : PostgreSQL (déploiement, compose) ou SQLite (poste sans Docker).

Les requêtes sont écrites avec des paramètres `%s` ; la couche SQLite les convertit en `?`.
L'URL choisit le moteur : `postgresql://...` ou `sqlite:///chemin/fichier.db`.
SQLite n'est qu'un repli de test local : les mesures et le dashboard visent PostgreSQL.
"""
import contextlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from bikeshare.config import ROOT

SCHEMA_FILE = ROOT / "sql" / "schema.sql"


class DatabaseUnavailable(RuntimeError):
    """Base injoignable ou erreur d'accès : l'appelant décide s'il peut continuer sans."""


class Database:
    def __init__(self, url):
        self.url = url
        self.dialect = "sqlite" if url.startswith("sqlite") else "postgresql"
        if self.dialect == "sqlite":
            self.path = url.split("sqlite:///", 1)[1]
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(self, sql, params):
        if self.dialect == "postgresql":
            return sql, params
        converted = []
        for value in params or ():
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, (dict, list)):
                value = json.dumps(value)
            converted.append(value)
        return sql.replace("%s", "?"), tuple(converted)

    @contextlib.contextmanager
    def connection(self):
        try:
            if self.dialect == "sqlite":
                conn = sqlite3.connect(self.path, timeout=30)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
            else:
                import psycopg
                from psycopg.rows import dict_row
                conn = psycopg.connect(self.url, row_factory=dict_row, connect_timeout=5)
        except Exception as exc:
            raise DatabaseUnavailable(f"base indisponible : {exc}") from exc
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql, params=()):
        sql, params = self._convert(sql, params)
        with self.connection() as conn:
            conn.cursor().execute(sql, params)

    def executemany_tx(self, statements):
        """Plusieurs écritures dans UNE transaction : tout ou rien."""
        with self.connection() as conn:
            cur = conn.cursor()
            for sql, params in statements:
                sql, params = self._convert(sql, params)
                cur.execute(sql, params)

    def query(self, sql, params=()):
        sql, params = self._convert(sql, params)
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def init_schema(self):
        script = SCHEMA_FILE.read_text(encoding="utf-8")
        with self.connection() as conn:
            if self.dialect == "sqlite":
                conn.executescript(script)
            else:
                conn.cursor().execute(script)


def from_env(env=None):
    """Base configurée par DATABASE_URL, ou None si absente (fonctionnalité alors désactivée)."""
    import os
    url = (env or os.environ).get("DATABASE_URL")
    return Database(url) if url else None


def record_pipeline_run(db, row):
    """Écrire (ou compléter) une ligne de `pipeline_runs`, clé = identifiant d'exécution."""
    columns = ["run_key", "batch_id", "status", "started_at", "finished_at", "duration_s",
               "failed_step", "error", "flow_run_id", "mlflow_run_id", "model_version",
               "qualification", "promoted"]
    values = [row.get(c) for c in columns]
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns[1:])
    db.execute(
        f"INSERT INTO pipeline_runs ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) "
        f"ON CONFLICT (run_key) DO UPDATE SET {updates}", values)
