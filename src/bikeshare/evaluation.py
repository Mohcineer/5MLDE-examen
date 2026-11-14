"""Métriques, validation croisée chronologique et comparaison à la baseline."""
import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from bikeshare.training import build_model, train_baseline


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate(X, y, params, n_splits):
    """CV chronologique (jamais de futur dans l'apprentissage) : moyenne et écart-type de la MAE."""
    maes = []
    for fit_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = build_model(params).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        maes.append(mean_absolute_error(y.iloc[val_idx], model.predict(X.iloc[val_idx])))
    return {"cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes))}


def evaluate(model, X_train, y_train, X_val, y_val, cv=None):
    """Toutes les métriques d'un candidat, baseline comprise, sur le MÊME lot de validation."""
    baseline = train_baseline(X_train, y_train)
    metrics = {}
    for prefix, X, y in (("train", X_train, y_train), ("val", X_val, y_val)):
        for key, value in regression_metrics(y, model.predict(X)).items():
            metrics[f"{prefix}_{key}"] = value
    for key, value in regression_metrics(y_val, baseline.predict(X_val)).items():
        metrics[f"baseline_val_{key}"] = value
    metrics["val_over_train_mae_ratio"] = metrics["val_mae"] / max(metrics["train_mae"], 1e-9)
    if cv:
        metrics.update(cv)
    return metrics
