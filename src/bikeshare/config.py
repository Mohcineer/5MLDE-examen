"""Chargement de la configuration (YAML) ; les secrets viennent de l'environnement."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_config(path=None):
    """Lire la configuration ; les chemins relatifs sont résolus depuis la racine du projet."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    for key in ("raw_dir", "batches_dir"):
        config["data"][key] = str((ROOT / config["data"][key]).resolve())
    return config
