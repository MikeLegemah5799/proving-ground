from __future__ import annotations

import pytest

from proving_ground.evaluation import HoldoutEval
from proving_ground.gates.base import GateContext
from proving_ground.gates.g1_g5 import (
    g1_data_validation,
    g2_metric_floors,
    g3_no_regression,
    g4_calibration,
    g5_slices,
)
from proving_ground.gates.g6_g8 import g6_performance, g8_model_card
from proving_ground.promote import train_candidate
from tests.conftest import make_toy_ws


@pytest.fixture
def ctx(tmp_path):
    ws = make_toy_ws(tmp_path)
    cand = train_candidate(ws, "ridge")
    b = ws.bundle()
    ho = b.holdout
    pred = ws.project.predict(cand.model, ho[b.features]).to_numpy()
    y, w = ho["y"].to_numpy(), ho["w"].to_numpy()
    sc = ws.project.score(y, pred, w)
    ev = HoldoutEval(cand.hash, b.data_version, y, w, pred, {"mse": sc["mse"], "r2": sc["r2"]}, {"mse": [sc["mse"] * .9, sc["mse"] * 1.1]},
                     {"calibration_error": 0.01}, {"cat": [{"level": "a", "n": 3000, "min_samples": 200, "weight": 3000.0, "scores": {"ae_ratio": 1.0}, "ci": {"ae_ratio": [0.97, 1.03]}}]})
    return GateContext(ws.project, b, cand, ev, ws.gates_cfg(), tmp_path / "r", seed=1, n_boot=50)


def test_g1_pass_and_leakage_fail(ctx):
    assert g1_data_validation(ctx).status == "pass"
    ctx.bundle.holdout.loc[0, "id"] = ctx.bundle.train.loc[0, "id"]
    r = g1_data_validation(ctx)
    assert r.status == "fail" and any(c.name == "split_leakage" and c.status == "fail" for c in r.checks)


def test_g1_schema_violation_fails(ctx):
    ctx.bundle.train.loc[0, "cat"] = "zzz"
    assert g1_data_validation(ctx).status == "fail"


def test_g2_pass_and_fail(ctx):
    assert g2_metric_floors(ctx).status == "pass"
    ctx.eval.metrics["r2"] = 0.1
    assert g2_metric_floors(ctx).status == "fail"


def test_g3_skip_without_champion_and_fail_on_regression(ctx):
    assert g3_no_regression(ctx).status == "skip"
    worse = HoldoutEval("x", ctx.eval.data_version, ctx.eval.y, ctx.eval.w, ctx.eval.pred + 1.0, {"mse": 3.0, "r2": 0.1}, {}, {}, {})
    ctx.champion, ctx.champion_eval = ctx.candidate, ctx.eval
    ctx.eval = worse
    assert g3_no_regression(ctx).status == "fail"


def test_g4_pass_and_fail(ctx):
    assert g4_calibration(ctx).status == "pass"
    ctx.eval.calibration["calibration_error"] = 0.5
    assert g4_calibration(ctx).status == "fail"


def test_g5_pass_warn_fail_and_insufficient(ctx):
    assert g5_slices(ctx).status == "pass"
    row = ctx.eval.slices["cat"][0]
    row["ci"]["ae_ratio"] = [0.8, 1.2]           # point estimate fine, CI exceeds tolerance -> warn
    assert g5_slices(ctx).status == "warn"
    row["scores"]["ae_ratio"] = 1.5              # beyond the gap -> fail
    assert g5_slices(ctx).status == "fail"
    row.update(n=10)                              # too small: never a silent pass
    r = g5_slices(ctx)
    assert r.status == "warn" and r.checks[0].value == "insufficient data"


def test_g6_budget_fail(ctx):
    assert g6_performance(ctx).status == "pass"
    ctx.cfg["g6_budget"]["p95_latency_ms"] = 0.0001
    assert g6_performance(ctx).status == "fail"


def test_g8_card_checks(ctx):
    ctx.card_text = None
    assert g8_model_card(ctx).status == "fail"
    ctx.card_text = f"# Card {ctx.candidate.hash}\n## Intended use\n"
    assert g8_model_card(ctx).status == "fail"
    ctx.card_text = f"# Card {ctx.candidate.hash}\n" + "".join(f"## {s}\nx\n" for s in ctx.cfg["g8_card"]["required_sections"])
    assert g8_model_card(ctx).status == "pass"
    ctx.card_text = ctx.card_text.replace(ctx.candidate.hash, "other")
    assert g8_model_card(ctx).status == "fail"


def test_g7_epsilon_and_failure(ctx):
    from proving_ground.gates.g6_g8 import g7_reproducibility
    assert g7_reproducibility(ctx).status == "warn"      # no retrain supplied: stated, not silent
    ctx.retrain = lambda: ctx.candidate
    assert g7_reproducibility(ctx).status == "pass"
