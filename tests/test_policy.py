from bikeshare.policy import PENDING, READY, REJECTED, beats_champion, decide


def good(policy):
    return {"protocol_id": policy["protocol_id"], "reload_ok": True,
            "metrics": {"val_mae": 30.0, "baseline_val_mae": 90.0, "val_r2": 0.9, "train_mae": 20.0,
                        "val_over_train_mae_ratio": 1.5}}


def test_accepted(policy):
    result = decide(good(policy), policy)
    assert result["status"] == READY and all(c["ok"] for c in result["checks"])


def test_rejected_when_baseline_not_beaten(policy):
    evidence = good(policy)
    evidence["metrics"]["baseline_val_mae"] = 31.0
    assert decide(evidence, policy)["status"] == REJECTED


def test_rejected_when_mae_too_high(policy):
    evidence = good(policy)
    evidence["metrics"]["val_mae"], evidence["metrics"]["baseline_val_mae"] = 150.0, 400.0
    assert decide(evidence, policy)["status"] == REJECTED


def test_rejected_when_overfitting(policy):
    evidence = good(policy)
    evidence["metrics"]["val_over_train_mae_ratio"] = 9.0
    assert decide(evidence, policy)["status"] == REJECTED


def test_rejected_when_reload_fails_or_unverified(policy):
    for reload_ok in (False, None):
        evidence = {**good(policy), "reload_ok": reload_ok}
        assert decide(evidence, policy)["status"] == REJECTED


def test_pending_on_other_protocol(policy):
    evidence = {**good(policy), "protocol_id": "autre"}
    assert decide(evidence, policy)["status"] == PENDING


def test_pending_on_non_finite_metric(policy):
    evidence = good(policy)
    evidence["metrics"]["val_mae"] = float("nan")
    assert decide(evidence, policy)["status"] == PENDING


def test_beats_champion(policy):
    assert beats_champion(25.0, 26.0, policy)["ok"]
    assert beats_champion(26.0, 26.0, policy)["ok"]          # égalité acceptée (min_gain = 0)
    assert not beats_champion(27.0, 26.0, policy)["ok"]
