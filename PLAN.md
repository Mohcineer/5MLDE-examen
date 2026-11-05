# Plan — Projet final 5MLDE : Bike Sharing

Sujet : prévoir le nombre de locations de vélos (régression). Données : UCI Bike Sharing (`hour.csv`, ~17 400 lignes).
Les performances ne sont pas notées ; on démontre une chaîne MLOps reproductible de bout en bout.

## 0. Décisions d'architecture (à justifier dans la synthèse)

| Choix | Décision | Pourquoi |
|---|---|---|
| Données | `hour.csv` rejoué **par lots** (ex. par semaine) pour simuler l'arrivée de nouvelles données | Jeu historique : à déclarer explicitement |
| Modèle | `GradientBoostingRegressor` ou `RandomForest` + baseline (moyenne par heure) | Simple, rechargeable, pipeline sklearn |
| Cible | `cnt` (locations par heure) ; métriques MAE, RMSE, R² vs baseline | Même logique que TP3 |
| Stack | Réutiliser `TP00/compose.yml` : MLflow, Prefect, PostgreSQL, Grafana | Déjà maîtrisée, exécution locale |
| API | FastAPI (prédiction unitaire + lot), validation Pydantic | Déjà vu au TP5 |
| Données du service | PostgreSQL : prédictions, feedbacks, erreurs, durées | Alimente Grafana |
| Charge | Locust ou k6 (à choisir selon TP5) | Comparer 2-3 configurations |
| Dépendances | `uv` + `pyproject.toml` + `uv.lock` | Reproductibilité |

## 1. Structure du dépôt (`EXAMEN/`)

```
EXAMEN/
├── README.md                  # sujet, architecture, données, commandes
├── SYNTHESE.md                # choix OPS, preuves, charge, dimensionnement, limites
├── pyproject.toml / uv.lock
├── compose.yml                # MLflow, Prefect, Postgres, Grafana, API
├── config/                    # config.yaml, policy.json (seuils), expectations
├── data/                      # (gitignoré) + scripts de téléchargement
├── src/bikeshare/
│   ├── ingestion.py           # téléchargement, découpage en lots, rejeu
│   ├── validation.py          # contrôles qualité bloquants
│   ├── preprocessing.py       # features déterministes
│   ├── training.py            # modèle + baseline
│   ├── evaluation.py          # métriques, rapport de décision
│   ├── registry.py            # versions, tags, alias champion, rollback
│   ├── policy.py              # fonction de décision testable
│   └── tracking.py            # helpers MLflow
├── flows/                     # Prefect : flow, deployment, automation d'alerte
├── api/                       # FastAPI, schémas, persistance, feedbacks
├── load_tests/                # scénarios et résultats
├── grafana/                   # dashboards provisionnés (JSON)
├── scripts/                   # demo_nominal.py, demo_incident.py
└── tests/                     # pytest : unitaires + intégration
```

## 2. Étapes, par critère d'évaluation

### Étape 1 — Socle code et reproductibilité (20 % + 10 %)
- Projet `uv`, config séparée (YAML + variables d'environnement), aucun notebook.
- Téléchargement reproductible du dataset + manifeste (URL, licence, SHA-256).
- Section « Données » du README : source UCI Bike Sharing, licence et conditions d'utilisation, citation, et mention explicite que le jeu historique est rejoué par lots pour simuler l'arrivée de nouvelles données.
- Tests `pytest` dès le départ, lint, README d'installation testé depuis un clone neuf.
- **Preuve :** `uv sync && uv run pytest` passent sur un poste propre.

### Étape 2 — Ingestion et qualité des données (20 % intégration, 20 % fiabilité)
- Découpage de `hour.csv` en lots chronologiques, rejeu par lot.
- Contrôles **bloquants** avant prétraitement : schéma, types, bornes (`hr` 0-23, `temp` dans [0,1]), doublons, valeurs manquantes, dérive (PSI) vs lot de référence.
- Échec = arrêt de l'entraînement, rapport écrit, aucun modèle publié.
- **Preuve :** un lot valide passe ; un lot corrompu volontairement est refusé.

### Étape 3 — Entraînement et traçabilité MLflow
- Un run par exécution : paramètres, métriques (train/val/CV, écart-type), empreinte et snapshot du lot, commit git, `log_input`.
- Modèle loggé avec signature et exemple d'entrée, rechargé dans un processus neuf.
- Baseline évaluée sur la même validation.
- **Preuve :** `runs:/<id>/model` se recharge seul.

### Étape 4 — Règles de mise en service et retour arrière
- `policy.json` versionnée et écrite avant les résultats : bat la baseline, MAE ≤ seuil, rechargement OK, métriques finies, protocole compatible, écart train/val borné.
- Fonction de décision testable → tag `qualification` (`pending` / `production-ready` / `rejected`) + rapport de décision en artefact.
- Commande `promote` : revérifie la politique, bat le champion actuel, pose l'alias `champion`.
- Commande `rollback` : repose `champion` sur la version précédente qualifiée.
- **Preuve :** un cas accepté, un cas refusé, un rollback.

### Étape 5 — Orchestration Prefect (20 % intégration, 20 % fiabilité)
- Flow `ingest → validate → preprocess → train → evaluate/publish`, tâches nommées.
- Retries bornés uniquement sur les erreurs transitoires ; pas de retry sur donnée invalide.
- Déploiement local planifié (`flow.serve`) qui traite le prochain lot automatiquement.
- Tags du flow run (batch_id, mlflow_run_id, étape en erreur) ; alerte automation sur `Failed`/`Crashed`.
- Alerte : Teams si possible, sinon destination HTTP de test (à déclarer).
- **Preuve :** exécution nominale, échec contrôlé, alerte reçue, correspondance batch → flow → run MLflow.

### Étape 6 — API de prédiction (20 % exploitation)
- `POST /predict` (unitaire) et `POST /predict/batch`, validation Pydantic, erreurs 422/503 claires.
- Charge le modèle via `models:/bikeshare@champion` ; la réponse contient `model_name` et `model_version`.
- `GET /health`, `GET /version`, rechargement à chaud ou au redémarrage après changement de `champion`.
- **Preuve :** prédiction avec version identifiée ; après `rollback`, la version change.

### Étape 7 — Persistance et feedbacks
- Tables PostgreSQL : `predictions` (entrée, sortie, version, durée, statut), `feedbacks` (valeur réelle, prediction_id), `errors`.
- `POST /feedback` avec la valeur réelle observée → MAE en ligne par version.
- **Preuve :** requêtes SQL montrant prédictions, versions, durées, erreurs.

### Étape 8 — Déploiement, tests de charge et dimensionnement (10 %)
- API **conteneurisée** (Dockerfile + service dans `compose.yml`) avec limites de ressources (`cpus`, `mem_limit`).
- Protocole fixé avant : scénario (mix unitaire/lot), montée en charge, durée, métriques (RPS, p50/p95/p99, taux d'erreur) et consommation CPU/RAM mesurée pendant la charge (`docker stats`).
- Comparer 2-3 configurations (limites CPU/RAM, workers uvicorn, taille du lot, modèle chargé en mémoire vs rechargé).
- Recommandation **chiffrée** : ressources nécessaires pour un objectif (ex. 50 RPS, p95 < 200 ms), avec marge et limites.
- **Preuve :** tableaux et courbes dans `SYNTHESE.md`.

### Étape 9 — Supervision Grafana
- Table PostgreSQL `pipeline_runs` alimentée par le flow à chaque exécution : lot, état, durée, étape et motif d'échec, `flow_run_id`, `mlflow_run_id`, version publiée. C'est la source des panneaux « traitements ».
- Datasource PostgreSQL, dashboards provisionnés en JSON dans Git.
- Panneaux : exécutions de traitements (succès/échec, durée), volume de requêtes, latence p50/p95, taux d'erreur, répartition par version, MAE en ligne via feedbacks.
- **Preuve :** dashboard avec données réelles d'une démo.

### Étape 10 — Démonstration et livrables
- `demo_nominal` : lot valide → entraînement → qualification → promote → prédiction → feedback → dashboard.
- `demo_incident` : lot invalide (ou MLflow/Postgres coupé) → blocage, alerte, aucun modèle publié, service intact sur l'ancien `champion`.
- README, `SYNTHESE.md` (décisions, preuves, charge, dimensionnement, limites), dépôt propre sans secrets ni gros fichiers.

## 3. Réutilisation des TP (à adapter, pas à recopier tel quel)

| Brique | Source |
|---|---|
| Compose, services | `TP00_setup_des_postes/compose.yml` |
| Orchestration, tags, alerte | `TP01_industrialisation/src/orchestration.py`, `setup_automation.py`, `setup_teams_block.py` |
| Registre, politique | `TP01_industrialisation/src/registry.py`, `policy.json` |
| API, DB, Grafana, charge | `TP05_api_supervision/` |

Le code est adapté au sujet Bike Sharing. Rien n'est modifié dans les dossiers de TP.

## 4. Ordre de réalisation recommandé

1. Socle + données + validation (étapes 1-2) : tout le reste en dépend.
2. Entraînement + MLflow + politique + rollback (3-4).
3. Prefect + alertes (5).
4. API + persistance + feedbacks (6-7).
5. Charge + Grafana (8-9).
6. Démos, README, synthèse (10).

Rédiger la synthèse au fil de l'eau : décisions et preuves se notent à chaque étape.

## 5. Risques et points d'attention

- **Rejeu par lots** : à déclarer explicitement (dataset historique).
- **Fuite de données** : découpage chronologique, prétraitement ajusté sur le train seulement.
- **Alerte Teams** : prévoir le repli HTTP de test dès le début.
- **Secrets** : URL webhook et identifiants hors Git (`.env` gitignoré, `.env.example` fourni).
- **Charge** : tester en local, sans exposer les services sur Internet ; dire que les mesures dépendent du poste.
- **Périmètre** : un modèle simple suffit ; ne pas passer de temps sur la performance prédictive.
