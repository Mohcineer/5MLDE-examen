"""Ingestion reproductible : téléchargement vérifié, puis rejeu du jeu historique par lots."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pandas as pd


class IngestionError(Exception):
    """Données absentes ou altérées : l'ingestion s'arrête explicitement."""


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def download_dataset(config, force=False):
    """Télécharger le zip UCI, vérifier son empreinte et en extraire `hour.csv`."""
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    csv_path = raw_dir / "hour.csv"
    if csv_path.exists() and not force:
        _check_csv(csv_path, data_cfg)
        return csv_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    response = httpx.get(data_cfg["source_url"], follow_redirects=True, timeout=60)
    response.raise_for_status()
    if sha256_bytes(response.content) != data_cfg["zip_sha256"]:
        raise IngestionError("Empreinte du zip différente de celle attendue : source modifiée ?")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_path.write_bytes(archive.read("hour.csv"))
    _check_csv(csv_path, data_cfg)
    return csv_path


def _check_csv(csv_path, data_cfg):
    if sha256_file(csv_path) != data_cfg["csv_sha256"]:
        raise IngestionError(f"Empreinte de {csv_path} inattendue : fichier altéré.")


def make_batches(csv_path, config):
    """Découper `hour.csv` en lots chronologiques de `batch_days` jours ; écrire un manifeste.

    Le jeu est historique : ce rejeu simule l'arrivée de nouvelles données.
    """
    batching = config["batching"]
    out_dir = Path(config["data"]["batches_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("instant")
    day_index = (pd.to_datetime(df["dteday"]) - pd.to_datetime(df["dteday"]).min()).dt.days
    df = df.assign(_batch=day_index // batching["batch_days"] + 1)

    manifest = []
    for batch_no, part in df.groupby("_batch"):
        part = part.drop(columns="_batch")
        path = out_dir / f"batch_{batch_no:04d}.csv"
        part.to_csv(path, index=False)
        manifest.append({
            "batch_id": f"batch_{batch_no:04d}",
            "file": path.name,
            "rows": len(part),
            "date_min": part["dteday"].min(),
            "date_max": part["dteday"].max(),
            "sha256": sha256_file(path),
            "role": "reference" if batch_no <= batching["reference_batches"] else "incoming",
        })
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_batch(config, batch_id):
    """Charger un lot et vérifier son empreinte contre le manifeste."""
    out_dir = Path(config["data"]["batches_dir"])
    manifest_path = out_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise IngestionError("Manifeste des lots absent : lancer scripts/prepare_data.py.")
    entries = {e["batch_id"]: e for e in json.loads(manifest_path.read_text(encoding="utf-8"))}
    if batch_id not in entries:
        raise IngestionError(f"Lot inconnu : {batch_id}")
    entry = entries[batch_id]
    path = out_dir / entry["file"]
    if not path.exists():
        raise IngestionError(f"Fichier du lot absent : {path}")
    if sha256_file(path) != entry["sha256"]:
        raise IngestionError(f"Empreinte du lot {batch_id} différente du manifeste.")
    return pd.read_csv(path)
