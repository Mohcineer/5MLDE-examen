"""Registre MLflow : enregistrement idempotent, qualification, mise en service (`champion`), retour arrière.

Règle centrale : l'alias `champion` n'est posé que si la politique COURANTE est revérifiée depuis les
preuves du run (jamais sur la foi d'un tag) ET si le candidat bat le champion en place sur le même
lot de validation. L'historique des promotions est conservé (pile de versions + journal) pour
permettre `rollback` sans intervention manuelle.
"""
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from bikeshare import tracking
from bikeshare.config import load_config, load_policy
from bikeshare.policy import READY, beats_champion, decide

EVIDENCE_FILE = "evaluation/evaluation.json"
STACK_TAG = "champion_stack"
LOG_TAG = "champion_log"
LOG_KEEP = 15


class PromotionRefused(Exception):
    """La version n'est pas éligible : l'alias n'est pas modifié."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def reload_check(model_uri, example):
    """Recharger le modèle dans un NOUVEAU processus et prédire (indépendant de l'état du parent)."""
    code = ("import sys, json, pandas as pd, mlflow;"
            "m = mlflow.sklearn.load_model(sys.argv[1]);"
            "print(len(m.predict(pd.DataFrame(json.loads(sys.argv[2])))))")
    done = subprocess.run([sys.executable, "-I", "-c", code, model_uri, example.to_json(orient="records")],
                          capture_output=True, text=True, timeout=300)
    return done.returncode == 0


def register(config, run_id, model_uri, batch_id, reload_ok):
    """Créer la version du modèle ; idempotent par `publication_id` (= run) pour supporter un retry."""
    client = tracking.client()
    name = config["model"]["registered_name"]
    if not client.search_registered_models(f"name = '{name}'"):
        client.create_registered_model(name)
    for version in client.search_model_versions(f"name = '{name}'"):
        if version.tags.get("publication_id") == run_id:
            return version
    return client.create_model_version(
        name, model_uri, run_id=run_id,
        tags={"publication_id": run_id, "batch_id": batch_id, "reload_ok": str(bool(reload_ok)).lower(),
              "qualification": "pending"})


def load_evidence(version):
    client = tracking.client()
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(version.run_id, EVIDENCE_FILE, tmp)
        evidence = json.loads(Path(local).read_text(encoding="utf-8"))
    evidence["reload_ok"] = version.tags.get("reload_ok") == "true"
    return evidence


def qualify(config, version_number, policy=None):
    """Évaluer la version selon la politique, taguer la VERSION et archiver un rapport de décision."""
    policy = policy or load_policy(config)
    client, name = tracking.client(), config["model"]["registered_name"]
    version = client.get_model_version(name, str(version_number))
    try:
        decision = decide(load_evidence(version), policy)
    except Exception as exc:  # preuve absente ou illisible : jamais de qualification positive
        decision = {"status": "pending", "checks": [], "reasons": [f"preuve illisible : {exc}"]}
    report = {"model": name, "version": str(version_number), "policy_id": policy["policy_id"],
              "date": _now(), **decision}
    client.log_dict(version.run_id, report, "evaluation/decision_report.json")
    for key, value in {"qualification": decision["status"], "qualification_policy": policy["policy_id"],
                       "qualification_date": report["date"],
                       "qualification_reason": "; ".join(decision["reasons"])[:500]}.items():
        client.set_model_version_tag(name, str(version_number), key, value)
    return report


def _read_json_tag(client, name, key, default):
    tags = client.get_registered_model(name).tags
    return json.loads(tags[key]) if key in tags else default


def current_champion(config):
    client, name = tracking.client(), config["model"]["registered_name"]
    try:
        return client.get_model_version_by_alias(name, config["model"]["champion_alias"])
    except Exception:
        return None


def _champion_mae_on(config, batch_id):
    """MAE du champion en place sur le lot `batch_id` (même lot que la validation du candidat)."""
    import mlflow
    from bikeshare.ingestion import load_batch
    from bikeshare.preprocessing import split_xy
    from sklearn.metrics import mean_absolute_error
    tracking.configure()
    model = mlflow.sklearn.load_model(f"models:/{config['model']['registered_name']}@"
                                      f"{config['model']['champion_alias']}")
    X, y = split_xy(load_batch(config, batch_id), config)
    return float(mean_absolute_error(y, model.predict(X)))


def _write_alias(client, config, name, version, action, actor, reason, stack, previous):
    client.set_registered_model_alias(name, config["model"]["champion_alias"], str(version))
    client.set_registered_model_tag(name, STACK_TAG, json.dumps(stack))
    log = _read_json_tag(client, name, LOG_TAG, [])
    log.append({"action": action, "version": str(version), "previous": previous, "actor": actor,
                "date": _now(), "reason": reason})
    client.set_registered_model_tag(name, LOG_TAG, json.dumps(log[-LOG_KEEP:]))


def promote(config, version_number, actor="cli", reason="", policy=None):
    """Poser `champion` sur la version si la politique courante ET la comparaison au champion passent."""
    policy = policy or load_policy(config)
    client, name = tracking.client(), config["model"]["registered_name"]
    version = client.get_model_version(name, str(version_number))
    decision = decide(load_evidence(version), policy)
    if decision["status"] != READY:
        raise PromotionRefused(f"v{version_number} non éligible : {decision['reasons']}")
    if version.tags.get("qualification") != READY or version.tags.get("qualification_policy") != policy["policy_id"]:
        raise PromotionRefused(f"v{version_number} non qualifiée avec la politique {policy['policy_id']} : "
                               "lancer `qualify` d'abord")
    previous = current_champion(config)
    comparison = None
    if previous is not None and previous.version == str(version_number):
        raise PromotionRefused(f"v{version_number} est déjà le champion.")
    if previous is not None:
        evidence = load_evidence(version)
        champion_mae = _champion_mae_on(config, evidence["batch_id"])
        comparison = beats_champion(evidence["metrics"]["val_mae"], champion_mae, policy)
        if not comparison["ok"]:
            raise PromotionRefused(f"v{version_number} ne bat pas le champion v{previous.version} : {comparison}")
    stack = _read_json_tag(client, name, STACK_TAG, [])
    stack.append(str(version_number))
    why = reason or (f"bat le champion v{previous.version}: {comparison}" if comparison else "premier champion")
    _write_alias(client, config, name, version_number, "promote", actor, why, stack,
                 previous.version if previous else None)
    return {"action": "promote", "version": str(version_number),
            "previous": previous.version if previous else None, "comparison": comparison}


def rollback(config, actor="cli"):
    """Remettre `champion` sur la version précédente encore qualifiée (dépile la version courante)."""
    client, name = tracking.client(), config["model"]["registered_name"]
    stack = _read_json_tag(client, name, STACK_TAG, [])
    current = current_champion(config)
    if current is None or len(stack) < 2:
        raise PromotionRefused("aucune version précédente à restaurer.")
    stack.pop()  # version courante
    while stack:
        target = stack[-1]
        tags = client.get_model_version(name, target).tags
        if tags.get("qualification") == READY:
            _write_alias(client, config, name, target, "rollback", actor,
                         f"retour arrière depuis v{current.version}", stack, current.version)
            return {"action": "rollback", "version": target, "previous": current.version}
        stack.pop()  # version devenue non qualifiée : on remonte plus loin
    raise PromotionRefused("aucune version précédente qualifiée.")


def status(config):
    client, name = tracking.client(), config["model"]["registered_name"]
    champion = current_champion(config)
    versions = sorted(client.search_model_versions(f"name = '{name}'"), key=lambda v: int(v.version))
    return {"champion": champion.version if champion else None,
            "versions": [{"version": v.version, "batch_id": v.tags.get("batch_id"),
                          "qualification": v.tags.get("qualification")} for v in versions],
            "log": _read_json_tag(client, name, LOG_TAG, []) if versions else []}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Qualifier, promouvoir ou restaurer une version")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("qualify", "promote"):
        p = sub.add_parser(cmd)
        p.add_argument("version")
        if cmd == "promote":
            p.add_argument("--reason", default="")
    sub.add_parser("rollback")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    config = load_config()
    try:
        if args.cmd == "qualify":
            out = qualify(config, args.version)
        elif args.cmd == "promote":
            out = promote(config, args.version, reason=args.reason)
        elif args.cmd == "rollback":
            out = rollback(config)
        else:
            out = status(config)
    except PromotionRefused as exc:
        print(f"REFUSÉ : {exc}")
        return 1
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
