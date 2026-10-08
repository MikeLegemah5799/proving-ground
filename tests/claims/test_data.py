from __future__ import annotations

import pandera.errors as pa_errors
import pytest
from examples.claims_frequency import data as D


def test_split_has_no_policy_or_profile_overlap_and_is_reproducible(ws):
    b = ws.bundle()
    parts = {"train": b.train, "val": b.val, "holdout": b.holdout, "stream": b.stream}
    ids = {k: set(v["IDpol"]) for k, v in parts.items()}
    names = list(parts)
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            assert not (ids[a] & ids[c]), (a, c)
    prof = {k: set(map(tuple, v[D.PROFILE].astype(str).to_numpy())) for k, v in parts.items()}
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            assert not (prof[a] & prof[c]), f"risk profile shared by {a} and {c}"
    assert sum(len(v) for v in parts.values()) == 40000
    d1 = D.split_groups(D.synthetic(5000, 1), seed=3)
    d2 = D.split_groups(D.synthetic(5000, 1), seed=3)
    assert all(d1[k].equals(d2[k]) for k in d1)


def test_cleaning_rules_report_counts_and_do_not_drop_rows():
    raw = D.synthetic(2000, 2)
    raw.loc[0, "ClaimNb"] = 9
    raw.loc[1, "Exposure"] = 1.5
    clean, rep = D.clean(raw)
    assert len(clean) == len(raw) and clean.ClaimNb.max() == 4 and clean.Exposure.max() == 1.0
    by = {r["rule"]: r["rows_affected"] for r in rep["rules"]}
    assert by["ClaimNb > 4"] == 1 and by["Exposure > 1"] == 1


def test_schema_rejects_bad_inputs(ws):
    schema = ws.project.schema()
    good = ws.bundle().reference[ws.bundle().features].head(5).copy()
    schema.validate(good)
    for col, val in [("DrivAge", 12), ("Exposure", 0.0), ("Exposure", 7.0), ("VehBrand", "B99"), ("Region", "R00")]:
        bad = good.copy()
        bad.loc[bad.index[0], col] = val
        with pytest.raises(pa_errors.SchemaErrors):
            schema.validate(bad, lazy=True)
    retyped = good.copy()
    retyped["BonusMalus"] = retyped["BonusMalus"].astype(str) + "%"
    with pytest.raises(pa_errors.SchemaErrors):
        schema.validate(retyped, lazy=True)


def test_small_slices_flagged_insufficient(ws):
    from proving_ground.evaluation import evaluate_holdout
    from proving_ground.gates.base import GateContext
    from proving_ground.gates.g1_g5 import g5_slices
    from proving_ground.promote import train_candidate
    c = train_candidate(ws, "glm")
    ev = evaluate_holdout(ws.project, ws.bundle(), c, ws.report_dir(c.hash), ws.reports_dir / "l.jsonl", 30, 1)
    ctx = GateContext(ws.project, ws.bundle(), c, ev, ws.gates_cfg(), ws.report_dir(c.hash))
    r = g5_slices(ctx)
    small = [ck for ck in r.checks if ck.value == "insufficient data"]
    assert small and all(ck.status == "warn" for ck in small)
