"""Télécharger UCI Bike Sharing et préparer les lots rejoués.

    uv run python scripts/prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bikeshare.config import load_config
from bikeshare.ingestion import download_dataset, make_batches


def main():
    config = load_config()
    csv_path = download_dataset(config)
    manifest = make_batches(csv_path, config)
    reference = sum(1 for e in manifest if e["role"] == "reference")
    print(f"{len(manifest)} lots écrits ({reference} de référence, {len(manifest) - reference} entrants).")


if __name__ == "__main__":
    main()
