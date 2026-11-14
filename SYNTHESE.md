# Synthèse — choix et résultats OPS

Projet : prévision du nombre de locations de vélos par heure (UCI Bike Sharing). Commandes dans
[README.md](README.md). Les performances prédictives ne sont pas évaluées ; les métriques servent aux
décisions de mise en service.

## 1. Décisions techniques

| Sujet | Décision | Justification |
|---|---|---|
| Données | `hour.csv` rejoué en 105 lots hebdomadaires (26 de référence) | Jeu historique, **déclaré comme tel** : le rejeu simule l'arrivée de données |
| Intégrité | SHA-256 du zip, du CSV et de chaque lot (manifeste) | Reproductibilité, détection d'altération |
| Qualité | Contrôles bloquants avant tout prétraitement ; dérive PSI en simple avertissement | La météo varie avec les saisons : PSI ≥ 0,25 sur `temp` pour 78 lots sur 78, bloquer rejetterait des lots légitimes. Limite : une dérive météo réelle n'est pas détectée |
| Modèle | GradientBoosting + encodage cyclique de l'heure ; baseline = moyenne par heure | Simple, rechargeable (pipeline sklearn unique) |
| Protocole | Walk-forward : lot N validé après entraînement sur les lots < N | Pas de fuite temporelle ; candidat et champion comparés sur le même lot |
| Politique | `config/policy.json` committée **avant** les premiers résultats | Les seuils ne sont pas ajustés a posteriori |
| Mise en service | Alias `champion` posé seulement si la politique est revérifiée depuis les preuves du run et si le candidat bat le champion ; rollback par pile de versions | Un tag saisi à la main ne suffit pas ; retour arrière sans réentraînement |
| Orchestration | Prefect : une tâche par étape ; 2 retries sur erreurs transitoires seulement | Une donnée invalide n'est jamais réessayée |
| Alerte | Automation Prefect sur `Failed`/`Crashed` → block webhook | **Déclaré** : validée ici avec un récepteur HTTP de test (`scripts/alert_receiver.py`) ; Teams est prévu (`--channel teams`, URL en saisie masquée) mais non testé ici |
| API | FastAPI, Pydantic (`extra=forbid`, bornes des contrôles qualité), version dans chaque réponse | Entrées validées, version tracée |
| Panne de base | `/predict` continue (trace perdue, comptée dans `/health`) ; `/feedback` renvoie 503 | Disponibilité de la prédiction ; un feedback non stocké n'a pas de sens |
| Persistance | Tables `pipeline_runs`, `api_requests` (4xx/5xx inclus), `predictions`, `feedbacks` | Alimente Grafana ; MAE en ligne par version |
| Charge | Générateur httpx maison | `locust` est bloqué sur ce poste (politique d'applications, DLL gevent) |

## 2. Preuves d'intégration (exécutées sur le poste de développement)

- **Nominal** (`scripts/demo_nominal.py --rollback`) : lots 27 à 32 traités ; versions 1 à 6
  `production-ready`, promues successivement ; l'API recharge le nouveau champion, sert 20 prédictions
  (v6) avec 20 feedbacks (MAE en ligne 50,4 sur la semaine suivante), puis suit le rollback v6 → v5. Exemple de promotion journalisée :
  « v2 bat le champion v1 (MAE 26,26 contre 31,31 sur le lot 28) ».
- **Refus** : promouvoir v1 après v3 est refusé (MAE 36,1 contre 17,4 sur le même lot) ; les tests
  couvrent une version rejetée (MAE aberrante) et une version sans preuves.
- **Rollback** : `rollback` remet `champion` de v4 sur v3 ; l'API, qui relit l'alias toutes les 10 s,
  passe de v6 à v5 sans redémarrage (constaté par la démo).
- **Traçabilité** : chaque run MLflow porte paramètres, métriques (train/val/CV, écart-type, baseline),
  `log_input` des jeux, snapshot du lot, commit git et état propre/modifié, SHA-256 de `uv.lock`, modèle
  avec signature et exemple, rechargement vérifié dans un processus neuf, rapport de décision.
- **Incident contrôlé** (`scripts/demo_incident.py`) : lot 31 corrompu (heure 99, humidité manquante,
  doublon). Résultat : flow `Failed` à l'étape `validate` sans retry ; run MLflow `FAILED` avec étape et
  rapport joints ; aucune nouvelle version, champion inchangé (v4) ; ligne `pipeline_runs` en échec ;
  **alerte reçue** avec `batch_id`, `mlflow_run_id` et `failed_step` dans les tags du flow run.
- **API et traces** : prédiction avec version, 422 détaillé (puis tracé), feedback 201/404/409, requêtes
  SQL donnant versions, durées et erreurs.
- **Tests** : 59 tests `pytest` (qualité des lots, modèle, politique, registre MLflow complet,
  base, API, rechargement à chaud).

## 3. Tests de charge

**Protocole** (fixé avant les mesures, `load_tests/run_campaign.py`). Mix 80 % `/predict`, 20 %
`/predict/batch` (20 éléments). Charge en modèle fermé, paliers de 1, 2, 4, 8, 16, 32 utilisateurs
simultanés, 20 s par palier après 5 s de chauffe. Mesures : débit, p50/p95/p99 côté client, taux
d'erreur, CPU et RSS du serveur (psutil). Serveur limité à `cores` cœurs par affinité CPU, générateur sur
d'autres cœurs. Poste : 8 cœurs logiques, 15 Go. Objectif de référence : **p95 < 200 ms**, 0 % d'erreur.

Débit (req/s) / p95 (ms) par concurrence — API avec traces écrites (SQLite) :

| Config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | RSS |
|---|---|---|---|---|---|---|---|
| 1 worker, 1 cœur | 46 / 29 | 54 / 53 | 67 / 124 | 76 / 265 | 80 / 731 | 68 / 2307 | 225 Mo |
| 2 workers, 2 cœurs | 40 / 33 | 51 / 58 | 81 / 85 | 97 / 180 | 94 / 707 | 90 / 1834 | 470 Mo |
| 4 workers, 4 cœurs | 49 / 25 | 55 / 51 | 81 / 81 | 105 / 134 | 107 / 357 | 104 / 1795 | 900 Mo |
| 2 w., lots de 5 | 43 / 27 | 57 / 49 | 83 / 78 | 104 / 171 | 101 / 643 | 100 / 1712 | 470 Mo |
| 2 w., lots de 100 | 42 / 41 | 50 / 62 | 57 / 144 | 69 / 301 | 67 / 1175 | 63 / 2740 | 470 Mo |
| 2 w., **sans traces** | 132 / 11 | 217 / 12 | 255 / 28 | 275 / 63 | 245 / 162 | 281 / 216 | 470 Mo |

Aucune erreur HTTP sur aucun palier.

**Lecture.**
1. Le goulot n'est pas le calcul mais l'**écriture des traces** : sans traces, 2 cœurs tiennent
   ~250 req/s à p95 < 30 ms ; avec traces le débit plafonne vers 80-105 req/s quel que soit le nombre de
   workers (1 → 4 cœurs ne gagne que ~30 %). Les écritures SQLite sont sérialisées.
2. Au-delà de la saturation, le débit ne monte plus et la latence explose (file d'attente) : le bon
   point de fonctionnement est avant le coude (c=4 à c=8).
3. La taille des lots pèse : lots de 100 → débit −30 % et p95 doublé par rapport aux lots de 5.
4. La mémoire croît avec les workers (~225 Mo chacun : le modèle et ses dépendances sont chargés par
   processus), pas avec la charge.

**Recommandation chiffrée (objectif 50 req/s, p95 < 200 ms, mix ci-dessus).**
- 1 worker, 1 cœur : tient 67 req/s à p95 124 ms, soit 50 req/s = 75 % de la capacité mesurée au dernier
  palier acceptable. Allouer **1 cœur et 512 Mo** (RSS 225 Mo + marge 2×).
- Pour la marge et la tolérance à la panne d'un processus : **2 workers, 2 cœurs, 1 Go** → 97 req/s à
  p95 180 ms (marge ×1,9 sur 50 req/s). Un 3ᵉ et 4ᵉ worker n'apportent presque rien tant que les traces
  sont écrites de façon synchrone.
- Au-delà de ~100 req/s : écrire les traces en différé (file/lot) ou sur PostgreSQL avec pool, avant
  d'ajouter des cœurs ; limiter les lots à ≤ 20 éléments.

## 4. Limites connues

- **Non exécutés sur le poste de développement** : Docker (`compose.yml`, `Dockerfile`, limites `cpus`/
  `mem_limit`), PostgreSQL (`sql/postgres_init.sql`, droits des rôles) et Grafana (dashboard JSON généré,
  14 panneaux, requêtes PostgreSQL) n'ont **pas été lancés ici**. Procédure de vérification sur une autre
  machine : README, dernière section. Le schéma SQL est testé sous SQLite uniquement.
- Les mesures de charge portent sur SQLite et des limites CPU par affinité (équivalent local de `cpus:`),
  sans `docker stats` ; 8 cœurs logiques partagent des cœurs physiques et le générateur tourne sur la même
  machine. Les chiffres sont des ordres de grandeur propres à ce poste ; sur PostgreSQL le plafond lié
  aux traces sera différent.
- Le serveur Prefect 3.8.7 est incompatible avec la version de FastAPI du projet : en local il tourne
  dans un environnement séparé (`scripts/local_stack.py`) ; l'image Docker officielle n'est pas concernée.
- Comparaison au champion : si le champion a déjà vu le lot de validation (cas d'une réactivation d'une
  ancienne version), elle lui est favorable ; en fonctionnement normal le lot est neuf pour les deux.
- Pas d'authentification sur l'API (usage local, ports liés à 127.0.0.1) ; alerte Teams non testée.
- La dérive n'est pas bloquante (choix justifié plus haut) ; modèle volontairement simple.
- Un dossier vide `C:\c\` a été créé par erreur sur le poste lors d'un chemin SQLite mal formé et n'a pas
  pu être supprimé (chemin protégé) ; il est sans effet.
