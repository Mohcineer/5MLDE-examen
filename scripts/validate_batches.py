"""Valider un ou tous les lots (rapports écrits dans data/reports).

    uv run python scripts/validate_batches.py              # tous les lots
    uv run python scripts/validate_batches.py batch_0030   # un lot
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bikeshare.config import load_config
from bikeshare.ingestion import load_batch
from bikeshare.validation import save_report, validate_batch


def main(batch_ids):
    config = load_config()
    manifest = Path(config["data"]["batches_dir"]) / "MANIFEST.json"
    import json
    all_ids = [e["batch_id"] for e in json.loads(manifest.read_text(encoding="utf-8"))]
    refused = 0
    for batch_id in batch_ids or all_ids:
        report = validate_batch(load_batch(config, batch_id), config, batch_id)
        save_report(report, config)
        if not report.passed:
            refused += 1
            print(f"REFUSÉ  {batch_id} : {[c.name for c in report.failures]}")
    print(f"{len(batch_ids or all_ids) - refused} lot(s) valide(s), {refused} refusé(s).")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
