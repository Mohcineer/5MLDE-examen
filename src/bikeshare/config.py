"""Chargement de la configuration (YAML) ; les secrets viennent de l'environnement."""
import json
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
    validation = config.get("validation", {})
    if "reports_dir" in validation:
        validation["reports_dir"] = str((ROOT / validation["reports_dir"]).resolve())
    return config


def load_policy(config=None):
    """Lire la politique de mise en service (versionnée dans Git, écrite avant les résultats)."""
    config = config or load_config()
    return json.loads((ROOT / config["tracking"]["policy_file"]).read_text(encoding="utf-8"))
