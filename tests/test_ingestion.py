import json

import pandas as pd
import pytest

from bikeshare.ingestion import IngestionError, load_batch, make_batches, sha256_file


@pytest.fixture
def config(tmp_path):
    return {
        "data": {"batches_dir": str(tmp_path / "batches")},
        "batching": {"batch_days": 7, "reference_batches": 2},
    }


@pytest.fixture
def csv_path(tmp_path):
    """Trois semaines de données horaires synthétiques (21 jours x 2 heures)."""
    days = pd.date_range("2011-01-01", periods=21).strftime("%Y-%m-%d")
    rows = [{"instant": i * 2 + h + 1, "dteday": day, "hr": h, "cnt": 10 + i + h}
            for i, day in enumerate(days) for h in (0, 1)]
    path = tmp_path / "hour.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_batches_are_chronological_and_complete(config, csv_path):
    manifest = make_batches(csv_path, config)
    assert [e["batch_id"] for e in manifest] == ["batch_0001", "batch_0002", "batch_0003"]
    assert sum(e["rows"] for e in manifest) == 42
    assert manifest[0]["date_max"] < manifest[1]["date_min"]


def test_roles_split_reference_and_incoming(config, csv_path):
    roles = [e["role"] for e in make_batches(csv_path, config)]
    assert roles == ["reference", "reference", "incoming"]


def test_load_batch_checks_fingerprint(config, csv_path):
    make_batches(csv_path, config)
    assert len(load_batch(config, "batch_0001")) == 14
    batch_file = next(__import__("pathlib").Path(config["data"]["batches_dir"]).glob("batch_0001.csv"))
    batch_file.write_text(batch_file.read_text() + "999,2011-01-01,0,1\n")
    with pytest.raises(IngestionError, match="Empreinte"):
        load_batch(config, "batch_0001")


def test_unknown_batch_is_refused(config, csv_path):
    make_batches(csv_path, config)
    with pytest.raises(IngestionError, match="inconnu"):
        load_batch(config, "batch_9999")


def test_manifest_is_valid_json(config, csv_path):
    make_batches(csv_path, config)
    path = __import__("pathlib").Path(config["data"]["batches_dir"]) / "MANIFEST.json"
    assert json.loads(path.read_text())[0]["sha256"] == sha256_file(
        path.parent / "batch_0001.csv")
