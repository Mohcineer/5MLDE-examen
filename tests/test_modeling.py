import numpy as np
import pytest

from bikeshare.evaluation import cross_validate, evaluate
from bikeshare.preprocessing import add_cyclic_features, history_before, split_xy
from bikeshare.training import HourMeanBaseline, train


def test_features_exclude_leaky_and_identifier_columns(config):
    forbidden = set(config["leaky_columns"]) | set(config["identifier_columns"]) | {config["target"]}
    assert not forbidden & set(config["features"]["raw"])


def test_split_xy_refuses_leaky_feature(config, frame):
    bad = {**config, "features": {"raw": config["features"]["raw"] + ["casual"]}}
    with pytest.raises(ValueError, match="interdites"):
        split_xy(frame, bad)


def test_cyclic_hour_is_continuous_around_midnight(frame):
    out = add_cyclic_features(frame[["hr"]])
    first, last, middle = out.iloc[0], out.iloc[23], out.iloc[12]
    assert np.hypot(first.hr_sin - last.hr_sin, first.hr_cos - last.hr_cos) < \
        np.hypot(first.hr_sin - middle.hr_sin, first.hr_cos - middle.hr_cos)


def test_history_is_strictly_chronological(frame):
    parts = {"batch_0001": frame.iloc[:48], "batch_0002": frame.iloc[48:96], "batch_0003": frame.iloc[96:]}
    history = history_before(parts, "batch_0003")
    assert len(history) == 96 and history["instant"].max() < parts["batch_0003"]["instant"].min()
    with pytest.raises(ValueError):
        history_before(parts, "batch_0001")


def test_baseline_predicts_hour_mean(frame):
    X, y = frame[["hr"]], frame["cnt"]
    model = HourMeanBaseline().fit(X, y)
    assert model.predict(frame[["hr"]].head(1))[0] == y[frame.hr == 0].mean()
    assert model.predict(frame[["hr"]].assign(hr=99).head(1))[0] == pytest.approx(y.mean())


def test_evaluate_reports_baseline_and_gap(config, frame):
    X, y = split_xy(frame, config)
    params = {**config["model"]["params"], "n_estimators": 20}
    model = train(X.iloc[:168], y.iloc[:168], params)
    cv = cross_validate(X.iloc[:168], y.iloc[:168], params, 2)
    metrics = evaluate(model, X.iloc[:168], y.iloc[:168], X.iloc[168:], y.iloc[168:], cv)
    for key in ("val_mae", "baseline_val_mae", "train_mae", "val_over_train_mae_ratio", "cv_mae_mean", "cv_mae_std"):
        assert np.isfinite(metrics[key])
