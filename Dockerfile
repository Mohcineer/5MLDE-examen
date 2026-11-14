# Image de l'API de prédiction (le modèle n'est PAS embarqué : il est chargé depuis MLflow par alias).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app

WORKDIR /app
# Dépendances d'abord (couche mise en cache tant que uv.lock ne change pas)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config ./config
COPY sql ./sql
COPY src ./src
COPY api ./api

ARG API_REVISION=unknown
ENV API_REVISION=${API_REVISION}
EXPOSE 8000
# Un seul processus uvicorn par défaut ; `API_WORKERS` est lu par compose.
CMD ["sh", "-c", "uv run --no-sync uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers ${API_WORKERS:-1}"]
