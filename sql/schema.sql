-- Schéma de traçabilité (PostgreSQL ; syntaxe volontairement compatible SQLite pour les tests locaux).
-- Réexécutable (IF NOT EXISTS). Les horodatages sont fournis par l'application (UTC).

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_key        text PRIMARY KEY,          -- flow_run_id Prefect, ou identifiant local
    batch_id       text NOT NULL,
    status         text NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    started_at     timestamptz NOT NULL,
    finished_at    timestamptz,
    duration_s     double precision,
    failed_step    text,
    error          text,
    flow_run_id    text,
    mlflow_run_id  text,
    model_version  text,
    qualification  text,
    promoted       integer                    -- 1 = devenu champion, 0 = non promu
);

CREATE TABLE IF NOT EXISTS api_requests (
    request_id             text PRIMARY KEY,
    started_at             timestamptz NOT NULL,
    endpoint               text NOT NULL,
    http_status            integer NOT NULL,
    item_count             integer,
    processing_duration_ms double precision NOT NULL,
    inference_duration_ms  double precision,
    model_version          text,
    error_code             text,              -- non nul si http_status >= 400
    campaign_id            text NOT NULL DEFAULT 'manual',
    worker_pid             integer,
    CHECK ((http_status < 400 AND error_code IS NULL) OR (http_status >= 400 AND error_code IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id  text PRIMARY KEY,
    request_id     text NOT NULL REFERENCES api_requests(request_id),
    item_index     integer NOT NULL,
    model_name     text NOT NULL,
    model_version  text NOT NULL,
    input_features jsonb NOT NULL,
    predicted_cnt  double precision NOT NULL,
    created_at     timestamptz NOT NULL,
    UNIQUE (request_id, item_index)
);

CREATE TABLE IF NOT EXISTS feedbacks (
    feedback_id    text PRIMARY KEY,
    prediction_id  text NOT NULL UNIQUE REFERENCES predictions(prediction_id),
    request_id     text REFERENCES api_requests(request_id),
    observed_cnt   double precision NOT NULL CHECK (observed_cnt >= 0),
    created_at     timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS api_requests_started_idx ON api_requests (started_at);
CREATE INDEX IF NOT EXISTS predictions_version_idx ON predictions (model_version, created_at);
CREATE INDEX IF NOT EXISTS pipeline_runs_started_idx ON pipeline_runs (started_at);
