from __future__ import annotations

import pytest

from proving_ground.promote import (
    PromotionRefused,
    evaluate_canary,
    evaluate_shadow,
    gate_candidate,
    promote,
    register_candidate,
    start_canary,
    start_shadow,
    train_candidate,
)
from proving_ground.rollback import rollback
from proving_ground.serving.app import attach_test_client
from proving_ground.traffic import run_traffic
from tests.conftest import make_toy_ws


def bootstrap_champion(ws):
    c = train_candidate(ws, "ridge")
    rep = gate_candidate(ws, c)
    assert rep["overall"] == "pass", rep["failed_gates"]
    register_candidate(ws, c.hash)
    promote(ws, c.hash)
    return c


def test_promotion_requires_passing_gate_report(toy_ws):
    c = train_candidate(toy_ws, "ridge")
    with pytest.raises(PromotionRefused, match="no gate report"):
        register_candidate(toy_ws, c.hash)
    with pytest.raises(PromotionRefused):
        promote(toy_ws, c.hash)


def test_degraded_candidate_rejected_and_audited(tmp_path):
    ws = make_toy_ws(tmp_path, degrade=True)
    c = train_candidate(ws, "ridge")
    rep = gate_candidate(ws, c)
    assert rep["overall"] == "fail" and rep["failed_gates"]
    with pytest.raises(PromotionRefused):
        register_candidate(ws, c.hash)
    ev = ws.audit.entries()
    assert ev[-1]["event"] == "reject" and ev[-1]["reason"].count("G")
    assert ws.audit.verify().ok


def test_full_cycle_and_rollback_idempotent(toy_ws):
    ws = toy_ws
    first = bootstrap_champion(ws)
    client = attach_test_client(ws)
    second = train_candidate(ws, "ridge", {"alpha": 2.0})
    assert second.hash != first.hash
    assert gate_candidate(ws, second)["overall"] == "pass"
    register_candidate(ws, second.hash)
    start_shadow(ws, second.hash)
    run_traffic(ws, client, n_windows=1, per_window=400, seed=1)
    sh = evaluate_shadow(ws, second.hash)
    assert sh["verdict"] == "pass", sh
    start_canary(ws, second.hash)
    run_traffic(ws, client, n_windows=1, per_window=600, seed=2, start_window=1)
    ca = evaluate_canary(ws, second.hash)
    assert ca["verdict"] == "pass", ca
    promote(ws, second.hash)
    assert client.get("/health").json()["champion_hash"] == second.hash
    out = rollback(ws, "drill")
    assert out["status"] == "rolled_back" and client.get("/health").json()["champion_hash"] == first.hash
    again = rollback(ws, "drill again")
    assert again["status"] == "already_rolled_back"
    assert [e["event"] for e in ws.audit.entries()].count("rollback") == 1
    assert ws.audit.verify().ok


def test_canary_aborts_on_error_spike(toy_ws):
    ws = toy_ws
    first = bootstrap_champion(ws)
    client = attach_test_client(ws)
    second = train_candidate(ws, "ridge", {"alpha": 2.0})
    gate_candidate(ws, second)
    register_candidate(ws, second.hash)
    start_shadow(ws, second.hash)
    run_traffic(ws, client, n_windows=1, per_window=400, seed=1)
    assert evaluate_shadow(ws, second.hash)["verdict"] == "pass"
    start_canary(ws, second.hash)
    client.post("/admin/fault", json={"candidate_error_rate": 0.5}, headers={"X-Admin-Token": "demo-token"}).raise_for_status()
    run_traffic(ws, client, n_windows=1, per_window=800, seed=3, start_window=1)
    ca = evaluate_canary(ws, second.hash)
    assert ca["verdict"] == "fail" and ca["aborted"]
    assert ws.audit.entries()[-1]["event"] == "canary_abort"
    assert ws.registry.state()["canary_pct"] == 0
    assert ws.registry.get_alias("champion") == first.hash
    with pytest.raises(PromotionRefused):
        promote(ws, second.hash)


def test_holdout_evaluated_once_per_candidate(toy_ws):
    ws = toy_ws
    c = train_candidate(ws, "ridge")
    gate_candidate(ws, c)
    gate_candidate(ws, c)
    lines = (ws.reports_dir / "holdout_ledger.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_gate_config_change_invalidates_report(toy_ws):
    ws = toy_ws
    c = train_candidate(ws, "ridge")
    gate_candidate(ws, c)
    ws.gates_path.write_text(ws.gates_path.read_text().replace("max: 0.6", "max: 0.7"))
    with pytest.raises(PromotionRefused, match="gates.yaml changed"):
        register_candidate(ws, c.hash)


def test_mlflow_tracking_is_optional_and_never_blocks(toy_ws, monkeypatch):
    pytest.importorskip("mlflow")
    monkeypatch.setenv("PG_TRACKING", "mlflow")
    toy_ws.tracking = "mlflow"
    c = train_candidate(toy_ws, "ridge")
    assert gate_candidate(toy_ws, c)["overall"] == "pass"
    register_candidate(toy_ws, c.hash)
    promote(toy_ws, c.hash)
    assert (toy_ws.root / "mlflow.db").exists()
