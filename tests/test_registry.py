"""Cycle registre complet sur un MLflow local SQLite (sans serveur) : qualifier, promouvoir, restaurer."""
import pytest

from bikeshare import registry, tracking
from bikeshare.evaluation import evaluate
from bikeshare.preprocessing import split_xy
from bikeshare.training import train


@pytest.fixture
def cfg(config, tmp_path):
    """Nom de modèle propre à chaque test : le registre MLflow est un état partagé du processus."""
    return {**config, "model": {**config["model"], "registered_name": f"bikeshare-{tmp_path.name}"}}


@pytest.fixture
def mlflow_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # les artefacts locaux (mlruns) restent dans le dossier de test
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")


def publish(config, policy, frame, batch_id, val_mae_shift=0.0):
    """Publier une version avec ses preuves, comme le fait step_evaluate."""
    X, y = split_xy(frame, config)
    params = {**config["model"]["params"], "n_estimators": 20}
    model = train(X, y, params)
    metrics = evaluate(model, X, y, X, y)
    metrics["val_mae"] += val_mae_shift
    metrics["baseline_val_mae"] = metrics["val_mae"] + 50
    metrics["val_over_train_mae_ratio"] = 1.2
    metrics["val_r2"] = 0.9
    run_id = tracking.open_run(config, batch_id)
    uri = tracking.log_model(run_id, model, X.head(3))
    reload_ok = registry.reload_check(uri, X.head(3))
    tracking.log_json(run_id, {"batch_id": batch_id, "protocol_id": policy["protocol_id"], "metrics": metrics,
                               "model_uri": uri, "reload_ok": reload_ok}, registry.EVIDENCE_FILE)
    version = registry.register(config, run_id, uri, batch_id, reload_ok)
    return version, reload_ok


def test_qualify_promote_refuse_and_rollback(cfg, policy, frame, mlflow_env, monkeypatch):
    v1, reload_ok = publish(cfg, policy, frame, "batch_0027")
    assert reload_ok                                      # preuve : rechargement dans un processus neuf
    assert registry.qualify(cfg, v1.version)["status"] == "production-ready"
    assert registry.promote(cfg, v1.version)["previous"] is None   # premier champion

    v2, _ = publish(cfg, policy, frame, "batch_0028")
    registry.qualify(cfg, v2.version)
    monkeypatch.setattr(registry, "_champion_mae_on", lambda cfg, batch: -1.0)  # champion imbattable
    with pytest.raises(registry.PromotionRefused, match="ne bat pas le champion"):
        registry.promote(cfg, v2.version)
    assert str(registry.current_champion(cfg).version) == str(v1.version)  # inchangé

    monkeypatch.setattr(registry, "_champion_mae_on", lambda cfg, batch: 1e6)  # champion mauvais
    assert registry.promote(cfg, v2.version)["version"] == str(v2.version)
    assert str(registry.current_champion(cfg).version) == str(v2.version)

    assert registry.rollback(cfg)["version"] == str(v1.version)
    assert str(registry.current_champion(cfg).version) == str(v1.version)
    with pytest.raises(registry.PromotionRefused):         # plus de version antérieure
        registry.rollback(cfg)


def test_unqualified_or_rejected_version_cannot_be_promoted(cfg, policy, frame, mlflow_env):
    version, _ = publish(cfg, policy, frame, "batch_0027", val_mae_shift=500.0)  # MAE aberrante
    assert registry.qualify(cfg, version.version)["status"] == "rejected"
    with pytest.raises(registry.PromotionRefused, match="non éligible"):
        registry.promote(cfg, version.version)
    assert registry.current_champion(cfg) is None


def test_register_is_idempotent_per_run(cfg, policy, frame, mlflow_env):
    version, _ = publish(cfg, policy, frame, "batch_0027")
    again = registry.register(cfg, version.run_id, version.source, "batch_0027", True)
    assert again.version == version.version
