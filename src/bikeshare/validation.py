"""Contrôles qualité d'un lot avant prétraitement.

Contrôles d'intégrité (schéma, types, valeurs manquantes, bornes, doublons, cohérence) :
BLOQUANTS, un échec lève `DataQualityError` et interrompt le pipeline.
Dérive (PSI) : AVERTISSEMENT, non bloquant (les saisons font varier la météo sans invalider un lot).
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PSI_CAP = 10.0  # valeur conventionnelle quand la référence est constante et le lot ne l'est plus


@dataclass
class CheckResult:
    name: str
    passed: bool
    blocking: bool
    detail: str = ""


@dataclass
class ValidationReport:
    batch_id: str
    rows: int
    checks: list = field(default_factory=list)
    created_at: str = ""

    @property
    def passed(self):
        """Valide si aucun contrôle bloquant n'a échoué."""
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def failures(self):
        return [c for c in self.checks if c.blocking and not c.passed]

    @property
    def warnings(self):
        return [c for c in self.checks if not c.blocking and not c.passed]

    def to_dict(self):
        return {"batch_id": self.batch_id, "rows": self.rows, "passed": self.passed,
                "created_at": self.created_at, "checks": [asdict(c) for c in self.checks]}


class DataQualityError(Exception):
    """Lot non conforme : l'entraînement ne doit pas démarrer."""

    def __init__(self, report):
        self.report = report
        names = ", ".join(c.name for c in report.failures)
        super().__init__(f"Lot {report.batch_id} non conforme ({names})")


def population_stability_index(reference, current, bins=10):
    """PSI entre deux séries ; classes définies par les quantiles de la référence."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # référence constante : pas de classes exploitables
        return 0.0 if np.all(current == reference[0]) else PSI_CAP
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-4
    p_ref = np.histogram(reference, edges)[0] / len(reference) + eps
    p_cur = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def validate_batch(df, config, batch_id="inconnu", reference=None):
    """Évaluer tous les contrôles ; ne lève rien (voir `ensure_valid`)."""
    cfg = config["validation"]
    report = ValidationReport(batch_id=batch_id, rows=len(df),
                              created_at=datetime.now(timezone.utc).isoformat())
    add = lambda name, passed, detail="", blocking=True: report.checks.append(
        CheckResult(name, bool(passed), blocking, detail))

    add("lignes_minimum", len(df) >= cfg["min_rows"], f"{len(df)} lignes (minimum {cfg['min_rows']})")

    expected = list(cfg["columns"]) + ["dteday"]
    missing = [c for c in expected if c not in df.columns]
    add("schema_colonnes", not missing, f"colonnes absentes : {missing}" if missing else "ok")
    if missing:  # sans le schéma, les contrôles suivants n'ont pas de sens
        return report

    not_numeric = [c for c in cfg["columns"] if not pd.api.types.is_numeric_dtype(df[c])]
    add("types_numeriques", not not_numeric, f"non numériques : {not_numeric}" if not_numeric else "ok")
    nulls = {c: int(n) for c, n in df[expected].isna().sum().items() if n}
    add("valeurs_manquantes", not nulls, f"{nulls}" if nulls else "ok")
    if not_numeric or nulls:
        return report

    for column, rule in cfg["columns"].items():
        bad = df[(df[column] < rule["min"]) | (df[column] > rule["max"])]
        if rule.get("integer"):
            bad = pd.concat([bad, df[df[column] % 1 != 0]]).drop_duplicates()
        add(f"bornes_{column}", bad.empty,
            f"{len(bad)} valeur(s) hors [{rule['min']}, {rule['max']}]" if len(bad) else "ok")

    dates = pd.to_datetime(df["dteday"], format="%Y-%m-%d", errors="coerce")
    add("dates_valides", dates.notna().all(), f"{int(dates.isna().sum())} date(s) invalide(s)")
    add("instant_unique", df["instant"].is_unique, f"{int(df['instant'].duplicated().sum())} doublon(s)")
    add("jour_heure_unique", not df.duplicated(["dteday", "hr"]).any(),
        f"{int(df.duplicated(['dteday', 'hr']).sum())} doublon(s) (jour, heure)")
    add("coherence_cnt", ((df["casual"] + df["registered"]) == df["cnt"]).all(),
        f"{int(((df['casual'] + df['registered']) != df['cnt']).sum())} ligne(s) avec casual+registered != cnt")

    if reference is not None and len(reference) and len(df):
        for column in cfg["drift_columns"]:
            psi = population_stability_index(reference[column], df[column])
            add(f"derive_{column}", psi < cfg["psi_warning"],
                f"PSI={psi:.3f} (seuil {cfg['psi_warning']})", blocking=False)
    return report


def save_report(report, config):
    """Écrire le rapport JSON ; sert de preuve d'un lot accepté ou refusé."""
    out_dir = Path(config["validation"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.batch_id}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def ensure_valid(df, config, batch_id="inconnu", reference=None):
    """Valider, écrire le rapport, et INTERROMPRE (exception) si un contrôle bloquant échoue."""
    report = validate_batch(df, config, batch_id, reference)
    save_report(report, config)
    if not report.passed:
        raise DataQualityError(report)
    return report
