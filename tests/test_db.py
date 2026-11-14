from datetime import datetime, timezone

import pytest

from bikeshare.db import Database, DatabaseUnavailable, record_pipeline_run


@pytest.fixture
def database(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'test.db'}")
    db.init_schema()
    return db


def test_schema_is_idempotent(database):
    database.init_schema()


def test_pipeline_run_is_upserted(database):
    now = datetime.now(timezone.utc)
    row = {"run_key": "r1", "batch_id": "batch_0030", "status": "running", "started_at": now}
    record_pipeline_run(database, row)
    record_pipeline_run(database, {**row, "status": "failed", "failed_step": "validate", "error": "bornes"})
    rows = database.query("SELECT status, failed_step FROM pipeline_runs")
    assert rows == [{"status": "failed", "failed_step": "validate"}]


def test_error_requests_require_error_code(database):
    now = datetime.now(timezone.utc)
    with pytest.raises(Exception):
        database.execute("INSERT INTO api_requests (request_id, started_at, endpoint, http_status, "
                         "processing_duration_ms) VALUES (%s, %s, %s, %s, %s)", ("a", now, "/predict", 500, 1.0))


def test_unreachable_postgres_raises_database_unavailable():
    with pytest.raises(DatabaseUnavailable):
        Database("postgresql://u:p@127.0.0.1:1/none").query("SELECT 1")
