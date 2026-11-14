"""Prétraitement déterministe : aucune statistique n'est apprise, donc aucune fuite possible.

Le modèle reçoit les colonnes brutes (`features.raw`) ; l'encodage cyclique de l'heure est la
première étape du pipeline sklearn. Le même code sert à l'entraînement et à l'API.
"""
import numpy as np
import pandas as pd


def add_cyclic_features(frame):
    """Ajouter hr_sin / hr_cos (l'heure 23 est voisine de l'heure 0)."""
    out = pd.DataFrame(frame).copy()
    angle = 2 * np.pi * out["hr"].astype(float) / 24.0
    out["hr_sin"] = np.sin(angle)
    out["hr_cos"] = np.cos(angle)
    return out


def history_before(batches, batch_id):
    """Lots strictement antérieurs à `batch_id`, concaténés dans l'ordre chronologique."""
    ordered = sorted(batches)
    previous = [batches[b] for b in ordered if b < batch_id]
    if not previous:
        raise ValueError(f"Aucun lot antérieur à {batch_id} : entraînement impossible.")
    return pd.concat(previous, ignore_index=True)


def split_xy(frame, config):
    """Séparer features brutes et cible ; les colonnes qui fuient la cible ne sont jamais utilisées."""
    raw = config["features"]["raw"]
    leaky = set(config["leaky_columns"]) | set(config["identifier_columns"])
    if leaky & set(raw):
        raise ValueError(f"Features interdites (fuite ou identifiant) : {sorted(leaky & set(raw))}")
    return frame[raw].copy(), frame[config["target"]].astype(float)
