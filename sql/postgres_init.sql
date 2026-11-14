-- Initialisation PostgreSQL (exécutée une fois par l'image postgres, en superutilisateur, base `bikeshare`).
-- Les mots de passe viennent des variables d'environnement du conteneur (\getenv) : rien dans Git.
\set ON_ERROR_STOP on
\getenv api_password API_DB_PASSWORD
\getenv grafana_password GRAFANA_DB_PASSWORD

CREATE ROLE api_writer LOGIN PASSWORD :'api_password';
CREATE ROLE grafana_reader LOGIN PASSWORD :'grafana_password';
ALTER ROLE grafana_reader SET statement_timeout = '10s';
ALTER ROLE grafana_reader SET default_transaction_read_only = on;
REVOKE ALL ON DATABASE bikeshare FROM PUBLIC;
GRANT CONNECT ON DATABASE bikeshare TO api_writer, grafana_reader;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO api_writer, grafana_reader;

\i /sql/schema.sql

-- L'historique d'utilisation est immuable pour l'API ; seul le suivi des traitements est mis à jour.
GRANT SELECT, INSERT ON api_requests, predictions, feedbacks TO api_writer;
GRANT SELECT, INSERT, UPDATE ON pipeline_runs TO api_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_reader;
