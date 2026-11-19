# Bikeshare MLOps — projet final 5MLDE

Chaîne MLOps complète pour prévoir le nombre de locations de vélos par heure (régression), de
l'ingestion à la supervision. Les performances prédictives ne sont pas l'objectif : on démontre une
chaîne opérationnelle, reproductible et maintenable. Choix, preuves, résultats de charge et limites :
[SYNTHESE.md](SYNTHESE.md). Plan de travail : [PLAN.md](PLAN.md).

## Architecture

```
 lots hebdo ─► Prefect (flow) ──► MLflow (runs, registre, alias champion)
 (rejeu)        ingest → validate → preprocess → train → evaluate → publish
                  │  échec ─► tags du flow run + automation ─► alerte (Teams / HTTP de test)
                  └─ écrit pipeline_runs ─┐
                                          ▼
 client ─► API FastAPI ─ charge models:/bikeshare@champion ─► PostgreSQL ◄── Grafana
           /predict /predict/batch /feedback   (traces : api_requests, predictions, feedbacks)
```

| Brique | Rôle | Où |
|---|---|---|
| `src/bikeshare/` | ingestion, validation, features, modèle, évaluation, politique, registre, pipeline | code métier, sans orchestrateur |
| `flows/` | flow Prefect, retries transitoires, alerte | orchestration |
| `api/` | API de prédiction, traces, feedbacks, rechargement à chaud | service |
| `sql/` | schéma (PostgreSQL, compatible SQLite) et initialisation | persistance |
| `grafana/` | datasource et dashboard provisionnés | supervision |
| `load_tests/` | générateur de charge, campagnes, résultats JSON | dimensionnement |
| `scripts/` | données, démos, pile locale, récepteur d'alertes | exploitation |
| `config/` | `config.yaml`, `policy.json` (règles de mise en service) | configuration |

## Données

- **Source** : UCI Machine Learning Repository, *Bike Sharing Dataset* (`hour.csv`, 17 379 lignes,
  2011-01-01 → 2012-12-31) — <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>.
- **Licence** : CC BY 4.0 (attribution requise).
- **Citation** : Fanaee-T, H. and Gama, J. (2013). *Event labeling combining ensemble detectors and
  background knowledge*. Progress in Artificial Intelligence.
- **Rejeu par lots** : le jeu est **historique**. Il est découpé en lots chronologiques de 7 jours
  (105 lots : 26 de référence, 79 « entrants ») pour **simuler** l'arrivée de nouvelles données.
- **Fuite de cible** : `casual + registered = cnt`. Ces colonnes ne sont jamais des features.
- **Intégrité** : zip et CSV vérifiés par SHA-256 (`config/config.yaml`) ; chaque lot est vérifié contre
  `data/batches/MANIFEST.json` au chargement. `data/` n'est pas versionné : il se régénère.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) ; Docker pour la pile de référence.

```sh
uv sync --python 3.12                      # dépendances figées par uv.lock
uv run python scripts/prepare_data.py      # télécharge, vérifie, découpe en lots
uv run pytest                              # tests automatisés
cp .env.example .env                       # puis remplacer les mots de passe (jamais dans Git)
```

## Lancer la pile

**Référence (Docker)** : MLflow, Prefect, PostgreSQL, Grafana et l'API, ressources bornées.

```sh
docker compose up -d --build               # Grafana : http://localhost:3000, Prefect : :4200, MLflow : :5000
export MLFLOW_TRACKING_URI=http://localhost:5000 PREFECT_API_URL=http://localhost:4200/api
export DATABASE_URL=postgresql://api_writer:<API_DB_PASSWORD>@localhost:5432/bikeshare
```

**Repli sans Docker** (poste restreint) : MLflow, Prefect, récepteur d'alertes et API en processus,
base SQLite à la place de PostgreSQL (pas de Grafana).

```sh
uv run python scripts/local_stack.py start      # puis : status / stop
```

## Utilisation

```sh
uv run python scripts/validate_batches.py              # contrôles qualité (rapports dans data/reports)
uv run python flows/setup_alerts.py --channel webhook  # automation d'alerte (ou --channel teams)
uv run python flows/pipeline_flow.py                   # traite le prochain lot
uv run python flows/pipeline_flow.py serve             # déploiement planifié (un lot / 5 min)
uv run python -m bikeshare.registry status|qualify N|promote N|rollback
uv run python scripts/demo_nominal.py --rollback       # démonstration nominale
uv run python scripts/demo_incident.py                 # incident contrôlé : lot corrompu
uv run python load_tests/run_campaign.py --all         # campagnes de charge
uv run python scripts/build_dashboard.py               # régénère le dashboard Grafana
```

## Pipeline et règles de mise en service

- **Protocole walk-forward** : pour le lot N, entraînement sur les lots < N, validation sur N. Le candidat
  et le champion en place sont comparés sur ce même lot. Baseline : moyenne de `cnt` par heure.
- **Contrôles bloquants** (`validation.py`) : lignes minimales, schéma, types, manquants, bornes, dates,
  unicité, cohérence `casual + registered = cnt`. Un échec lève `DataQualityError` : aucun modèle publié.
  La dérive (PSI sur `hr`, `weathersit`) est un avertissement : la météo varie avec les saisons.
- **Politique** (`config/policy.json`, écrite avant les résultats) : bat la baseline de ≥ 5 en MAE,
  MAE ≤ 110, R² ≥ 0,3, rapport MAE val/train ≤ 5, rechargement dans un processus neuf, métriques finies,
  protocole compatible. Fonction pure `policy.decide` → tag `qualification` :
  `production-ready` / `pending` / `rejected`.
- **Promotion** : revérifie la politique depuis les preuves du run, puis exige de battre le champion actuel
  sur le même lot. **Rollback** : dépile la dernière promotion (pile et journal dans les tags du modèle).
- **Orchestration** : retries bornés (2) uniquement sur erreurs transitoires (MLflow, base, réseau),
  jamais sur une donnée invalide. Tags du flow run : `batch_id`, `mlflow_run_id`, `failed_step`.

## API

| Endpoint | Rôle |
|---|---|
| `POST /predict` | une prédiction ; réponse avec `model_name`, `model_version`, `prediction_id` |
| `POST /predict/batch` | 1 à 100 prédictions |
| `POST /feedback` | valeur réelle observée pour un `prediction_id` (201, 404, 409) |
| `GET /health`, `GET /version` | état, version et run MLflow servis |

Entrées validées par Pydantic (422 détaillé), modèle indisponible → 503. Le champion est relu toutes
les 10 s : après `promote` ou `rollback`, la version servie change sans redémarrage.

## Vérification de la pile Docker

Le poste de développement n'a pas Docker. La pile complète a été vérifiée le 16/11/2026 sur un second
poste (Windows, Docker Desktop), à partir d'un clone du dépôt. Résultats : SYNTHESE.md, section 2.

1. `docker compose up -d --build` puis `docker compose ps` : les 5 services *healthy* (l'API passe
   *healthy* une fois qu'un premier champion existe, car sa sonde interroge `/version`).
2. `uv run python scripts/prepare_data.py`, puis `uv run python scripts/demo_nominal.py --rollback`
   et `uv run python scripts/demo_incident.py` (récepteur d'alertes lancé, automation créée avec
   `ALERT_WEBHOOK_URL=http://host.docker.internal:8765/alert` : Prefect tourne dans un conteneur).
3. Grafana (`http://localhost:3000`, dossier *Bikeshare*) : panneaux alimentés par PostgreSQL.
4. Charge sous Docker : `uv run python load_tests/run_campaign.py --docker http://127.0.0.1:8000
   --name docker-w1-c1` mesure l'API du conteneur (CPU et mémoire via `docker stats`, traces dans
   PostgreSQL). Pour comparer des configurations, changer `API_CPUS` / `API_WORKERS` / `API_MEM_LIMIT`
   dans `.env`, puis `docker compose up -d api` (recrée l'API) et relancer avec un autre `--name`.
   Résultats dans `load_tests/results/`.

**Dépannage.** `port is already allocated` : un autre conteneur occupe 3000, 4200, 5000 ou 5432
(`docker ps`, puis `docker stop`). Si un service a été créé pendant ce conflit, le recréer
(`docker compose up -d --force-recreate <service>`). Les scripts ne lisent pas `.env` : exporter
`MLFLOW_TRACKING_URI`, `PREFECT_API_URL`, `DATABASE_URL` et `ALERT_WEBHOOK_URL` dans le terminal.
