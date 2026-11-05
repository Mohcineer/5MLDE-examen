import json

import numpy as np
import pandas as pd
import pytest

from bikeshare.config import load_config
from bikeshare.validation import (DataQualityError, ensure_valid, population_stability_index,
                                  validate_batch)


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg["validation"]["reports_dir"] = str(tmp_path / "reports")
    return cfg


@pytest.fixture
def batch():
    """Deux jours de données horaires valides."""
    rng = np.random.default_rng(0)
    rows = []
    for day in (0, 1):
        for hr in range(24):
            casual, registered = int(rng.integers(0, 30)), int(rng.integers(0, 150))
            rows.append({
                "instant": day * 24 + hr + 1, "dteday": f"2011-01-0{day + 1}", "season": 1,
                "yr": 0, "mnth": 1, "hr": hr, "holiday": 0, "weekday": 6, "workingday": 0,
                "weathersit": 1, "temp": 0.24, "atemp": 0.28, "hum": 0.8, "windspeed": 0.1,
                "casual": casual, "registered": registered, "cnt": casual + registered})
    return pd.DataFrame(rows)


def failed(report):
    return {c.name for c in report.failures}


def test_valid_batch_passes(batch, config):
    report = validate_batch(batch, config, "b1")
    assert report.passed and not report.failures


def test_missing_column_blocks(batch, config):
    report = validate_batch(batch.drop(columns="temp"), config)
    assert failed(report) == {"schema_colonnes"}


def test_null_blocks(batch, config):
    batch.loc[3, "hum"] = np.nan
    assert "valeurs_manquantes" in failed(validate_batch(batch, config))


def test_non_numeric_blocks(batch, config):
    batch["temp"] = batch["temp"].astype(str).replace("0.24", "chaud")
    assert "types_numeriques" in failed(validate_batch(batch, config))


@pytest.mark.parametrize("column,value", [("hr", 25), ("temp", 1.5), ("weathersit", 7), ("cnt", -3)])
def test_out_of_bounds_blocks(batch, config, column, value):
    batch.loc[2, column] = value
    batch.loc[2, "casual"], batch.loc[2, "registered"] = 0, 0
    assert f"bornes_{column}" in failed(validate_batch(batch, config))


def test_non_integer_hour_blocks(batch, config):
    batch["hr"] = batch["hr"].astype(float)
    batch.loc[5, "hr"] = 5.5
    assert "bornes_hr" in failed(validate_batch(batch, config))


def test_duplicates_block(batch, config):
    duplicated = pd.concat([batch, batch.iloc[[0]]], ignore_index=True)
    assert {"instant_unique", "jour_heure_unique"} <= failed(validate_batch(duplicated, config))


def test_inconsistent_total_blocks(batch, config):
    batch.loc[0, "cnt"] += 5
    assert "coherence_cnt" in failed(validate_batch(batch, config))


def test_invalid_date_blocks(batch, config):
    batch.loc[0, "dteday"] = "pas-une-date"
    assert "dates_valides" in failed(validate_batch(batch, config))


def test_too_few_rows_blocks(batch, config):
    assert "lignes_minimum" in failed(validate_batch(batch.head(5), config))


def test_ensure_valid_raises_and_writes_report(batch, config):
    batch.loc[0, "hr"] = 99
    with pytest.raises(DataQualityError) as exc:
        ensure_valid(batch, config, "batch_bad")
    assert "bornes_hr" in str(exc.value)
    saved = json.loads((__import__("pathlib").Path(config["validation"]["reports_dir"])
                        / "batch_bad.json").read_text())
    assert saved["passed"] is False


def test_ensure_valid_returns_report_when_ok(batch, config):
    assert ensure_valid(batch, config, "batch_ok").passed


def test_drift_is_a_warning_not_a_blocker(batch, config):
    reference = batch.copy()
    shifted = batch.copy()
    shifted["weathersit"] = 3
    report = validate_batch(shifted, config, "b", reference=reference)
    assert report.passed                                   # non bloquant
    assert any(c.name == "derive_weathersit" for c in report.warnings)


def test_psi_identical_is_zero_and_shift_is_large():
    values = np.random.default_rng(1).normal(size=500)
    assert population_stability_index(values, values) < 0.01
    assert population_stability_index(values, values + 3) > 0.25
