import pandas as pd
import pytest

from bikeshare.config import load_config, load_policy


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def policy(config):
    return load_policy(config)


@pytest.fixture
def frame():
    """Petit jeu synthétique au schéma du lot."""
    rows = []
    for day in range(10):
        for hr in range(24):
            rows.append({"instant": day * 24 + hr + 1, "dteday": f"2011-01-{day + 1:02d}", "season": 1, "yr": 0,
                         "mnth": 1, "hr": hr, "holiday": 0, "weekday": day % 7, "workingday": 1,
                         "weathersit": 1, "temp": 0.3, "atemp": 0.3, "hum": 0.5, "windspeed": 0.1,
                         "casual": 5, "registered": 20 + 5 * (6 <= hr <= 20), "cnt": 25 + 5 * (6 <= hr <= 20)})
    return pd.DataFrame(rows)
