"""Règles de mise en service : fonction pure, sans MLflow ni effet de bord.

`decide(evidence, policy)` consomme des résultats déjà calculés. Elle retourne un statut :
`production-ready` (tout est satisfait), `rejected` (une règle de qualité échoue) ou `pending`
(preuve manquante ou protocole incompatible : on ne sait pas, donc on ne qualifie pas).
"""
import math

PENDING, READY, REJECTED = "pending", "production-ready", "rejected"


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check(name, ok, observed, expected):
    return {"check": name, "ok": bool(ok), "observed": observed, "expected": expected}


def decide(evidence, policy):
    """evidence : {"protocol_id", "metrics" (dict), "reload_ok" (bool | None)}."""
    metrics = evidence.get("metrics") or {}
    checks, pending, rejected = [], [], []

    same = evidence.get("protocol_id") == policy["protocol_id"]
    checks.append(_check("same_protocol", same, evidence.get("protocol_id"), policy["protocol_id"]))
    if not same:
        pending.append("protocole de validation incompatible avec la politique")

    primary, secondary = policy["primary"], policy["secondary"]
    needed = [f"val_{primary['metric']}", f"baseline_val_{primary['metric']}",
              f"val_{secondary['metric']}", "train_mae", "val_over_train_mae_ratio"]
    finite = all(_finite(metrics.get(key)) for key in needed)
    checks.append(_check("metrics_finite", finite, {k: metrics.get(k) for k in needed}, "finies"))
    if not finite:
        pending.append("métrique manquante ou non finie")

    reload_ok = evidence.get("reload_ok")
    checks.append(_check("reload_ok", reload_ok is True, reload_ok, True))
    if reload_ok is not True:
        rejected.append("modèle non rechargeable dans un processus neuf (ou non vérifié)")

    if finite:
        mae = metrics[f"val_{primary['metric']}"]
        base = metrics[f"baseline_val_{primary['metric']}"]
        gain = base - mae
        ok = gain >= primary["min_gain_vs_baseline"]
        checks.append(_check("mae_gain_vs_baseline", ok, round(gain, 4), f">= {primary['min_gain_vs_baseline']}"))
        if not ok:
            rejected.append("ne bat pas suffisamment la baseline (MAE)")
        ok = mae <= primary["max_absolute"]
        checks.append(_check("mae_absolute", ok, round(mae, 4), f"<= {primary['max_absolute']}"))
        if not ok:
            rejected.append("MAE trop élevée pour l'usage visé")
        r2 = metrics[f"val_{secondary['metric']}"]
        ok = r2 >= secondary["min"]
        checks.append(_check("r2_min", ok, round(r2, 4), f">= {secondary['min']}"))
        if not ok:
            rejected.append("R² insuffisant")
        ratio = metrics["val_over_train_mae_ratio"]
        ok = ratio <= policy["max_val_over_train_mae_ratio"]
        checks.append(_check("train_val_gap", ok, round(ratio, 3), f"<= {policy['max_val_over_train_mae_ratio']}"))
        if not ok:
            rejected.append("écart train/validation trop grand (sur-apprentissage)")

    if rejected:
        return {"status": REJECTED, "reasons": rejected + pending, "checks": checks}
    if pending:
        return {"status": PENDING, "reasons": pending, "checks": checks}
    return {"status": READY, "reasons": ["tous les contrôles de la politique sont satisfaits"],
            "checks": checks}


def beats_champion(candidate_mae, champion_mae, policy):
    """Le candidat doit faire au moins aussi bien que le champion sur le même lot de validation."""
    gain = champion_mae - candidate_mae
    return {"ok": gain >= policy["min_gain_vs_champion"], "gain": round(gain, 4),
            "candidate_mae": round(candidate_mae, 4), "champion_mae": round(champion_mae, 4)}
