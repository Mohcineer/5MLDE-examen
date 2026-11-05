# Bikeshare MLOps — projet final 5MLDE

Chaîne MLOps complète pour prévoir le nombre de locations de vélos par heure (régression),
de l'ingestion à la supervision. Les performances prédictives ne sont pas l'objectif : on démontre
une chaîne opérationnelle, reproductible et maintenable. Le plan de travail est dans [PLAN.md](PLAN.md).

> État : étape 1 (socle, données) en cours. Les autres sections seront complétées au fil du projet.

## Données

- **Source** : UCI Machine Learning Repository, *Bike Sharing Dataset* (`hour.csv`, 17 379 lignes,
  2011-01-01 → 2012-12-31) — <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>.
- **Licence** : CC BY 4.0 (attribution requise).
- **Citation** : Fanaee-T, H. and Gama, J. (2013). *Event labeling combining ensemble detectors and
  background knowledge*. Progress in Artificial Intelligence.
- **Rejeu par lots** : le jeu est **historique**. Il est découpé en lots chronologiques de 7 jours
  (105 lots : 26 de référence, 79 « entrants ») pour **simuler** l'arrivée de nouvelles données.
- **Fuite de cible** : `casual + registered = cnt`. Ces colonnes ne sont jamais utilisées comme features.
- **Intégrité** : le zip et le CSV sont vérifiés par SHA-256 (voir `config/config.yaml`) ; chaque lot
  est vérifié contre `data/batches/MANIFEST.json` au chargement.
- Les données ne sont pas versionnées (`data/` est ignoré) : elles se régénèrent par la commande ci-dessous.

## Contrôles qualité (avant prétraitement)

`src/bikeshare/validation.py`, bornes dans `config/config.yaml`. Un rapport JSON par lot est écrit dans
`data/reports/`.

| Contrôle | Effet |
|---|---|
| Nombre de lignes minimum, schéma, types numériques, valeurs manquantes | **Bloquant** |
| Bornes de chaque colonne (dont entiers), dates valides | **Bloquant** |
| Unicité de `instant` et de (`dteday`, `hr`), cohérence `casual + registered = cnt` | **Bloquant** |
| Dérive PSI vs lots de référence (`hr`, `weathersit`) | Avertissement |

Un échec bloquant lève `DataQualityError` : l'entraînement ne démarre pas.

**Choix sur la dérive.** Elle est non bloquante : sur des lots hebdomadaires comparés à 6 mois de référence,
la météo varie avec les saisons (PSI ≥ 0,25 sur `temp` pour 78 lots entrants sur 78), ce qui rejetterait des
lots légitimes. Seules `hr` et `weathersit`, stables (0/78 et 15/78 lots signalés), sont suivies. Limite :
une dérive réelle de la météo n'est donc pas détectée par ce contrôle.

```sh
uv run python scripts/validate_batches.py [batch_0030]   # valide tous les lots ou un lot
```

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/), Docker (pour les services, étapes suivantes).

```sh
uv sync --python 3.12          # dépendances figées par uv.lock
uv run python scripts/prepare_data.py   # télécharge, vérifie, découpe en lots
uv run pytest                  # tests automatisés
```

Copier `.env.example` vers `.env` pour les variables d'environnement (aucun secret dans Git).
