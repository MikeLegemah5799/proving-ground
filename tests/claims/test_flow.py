from __future__ import annotations

import pytest

from proving_ground.monitoring.matrix import run_matrix
from proving_ground.promote import (
    PromotionRefused,
    gate_candidate,
    promote,
    register_candidate,
    train_candidate,
)
from proving_ground.serving.app import attach_test_client
from tests.claims.conftest import claims_ws

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cws(tmp_path_factory):
    w = claims_ws(tmp_path_factory.mktemp("flow"))
    w.bundle()
    return w


def test_determinism_same_seed_same_hash(cws):
    a = train_candidate(cws, "glm")
    b = train_candidate(cws, "glm")
    assert a.hash == b.hash
    g1, g2 = train_candidate(cws, "gbm"), train_candidate(cws, "gbm")
    assert g1.hash == g2.hash


def test_champion_then_degraded_challenger_rejected_with_named_gate(cws):
    champ = train_candidate(cws, "glm")
    rep = gate_candidate(cws, champ)
    assert rep["overall"] == "pass", [(g["id"], [c for c in g["checks"] if c["status"] == "fail"]) for g in rep["gates"] if g["status"] == "fail"]
    register_candidate(cws, champ.hash)
    promote(cws, champ.hash)
    bad = train_candidate(cws, "gbm_degraded")
    r = gate_candidate(cws, bad)
    assert r["overall"] == "fail" and r["failed_gates"]
    with pytest.raises(PromotionRefused):
        register_candidate(cws, bad.hash)
    last = cws.audit.entries()[-1]
    assert last["event"] == "reject" and any(g in last["reason"] for g in r["failed_gates"])
    assert cws.audit.verify().ok


def test_model_card_is_regenerated_per_candidate_with_all_sections(cws):
    required = cws.gates_cfg()["g8_card"]["required_sections"]
    cards = {}
    for fam in ("glm", "gbm"):
        c = train_candidate(cws, fam)
        gate_candidate(cws, c)
        txt = (cws.report_dir(c.hash) / "MODEL_CARD.md").read_text()
        assert c.hash in txt
        for s in required:
            assert f"## {s}" in txt, s
        assert "no timestamps" in txt.lower() and "proxy" in txt.lower() and "Synthetic data" in txt
        cards[fam] = txt
    assert cards["glm"] != cards["gbm"]


@pytest.fixture(scope="module")
def matrix(cws):
    train = [c for c in [train_candidate(cws, "glm")]]
    if cws.registry.get_alias("champion") is None:
        gate_candidate(cws, train[0])
        register_candidate(cws, train[0].hash)
        promote(cws, train[0].hash)
    return run_matrix(cws, per_window=1500)


def test_every_scenario_triggers_its_declared_outcome(matrix):
    by = {r["scenario"]: r for r in matrix["scenarios"]}
    assert len(by) >= 9
    for name, r in by.items():
        assert r["synthetic"] is True
        assert r["detected"], f"{name}: {r['note']} {r['window_levels']} {r['signal_levels']}"
    assert by["silent_decay"]["watch_before_alert"]
    assert by["schema_break"]["signal_levels"].count("alert") >= 1


def test_schema_break_is_counted_by_serving(cws):
    client = attach_test_client(cws)
    b = cws.bundle()
    rec = {c: (v.item() if hasattr(v, "item") else v) for c, v in b.reference[b.features].iloc[0].items()}
    rec["key"] = "x1"
    rec["BonusMalus"] = "77%"
    r = client.post("/predict", json={"records": [rec], "synthetic": True}).json()
    assert r["predictions"][0]["error"]["type"] == "validation"
    assert client.get("/metrics").json()["invalid"] >= 1
