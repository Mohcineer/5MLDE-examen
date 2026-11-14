"""Modèle et baseline.

Modèle : GradientBoostingRegressor derrière l'encodage cyclique de l'heure (pipeline sklearn unique,
donc rechargeable tel quel). Baseline : moyenne de `cnt` par heure de la journée.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bikeshare.preprocessing import add_cyclic_features


class HourMeanBaseline(BaseEstimator, RegressorMixin):
    """Prédit la moyenne historique de `cnt` pour l'heure demandée (repli : moyenne globale)."""

    def fit(self, X, y):
        table = pd.Series(np.asarray(y, float), index=np.asarray(X["hr"])).groupby(level=0).mean()
        self.table_ = table.to_dict()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.array([self.table_.get(h, self.global_mean_) for h in np.asarray(X["hr"])])


def build_model(params):
    return Pipeline([
        ("features", FunctionTransformer(add_cyclic_features, validate=False)),
        ("regressor", GradientBoostingRegressor(**params)),
    ])


def train(X, y, params):
    return build_model(params).fit(X, y)


def train_baseline(X, y):
    return HourMeanBaseline().fit(X, y)
