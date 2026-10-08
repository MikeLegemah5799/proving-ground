from __future__ import annotations

import numpy as np

from ..stats import ci as _ci
from ..stats import paired_bootstrap
from .base import FAIL, PASS, SKIP, WARN, Check, GateContext, GateResult


def g1_data_validation(ctx: GateContext) -> GateResult:
    cfg, b = ctx.cfg.get("g1_data", {}), ctx.bundle
    r = GateResult("G1", "Data validation")
    schema = ctx.project.schema()
    splits = {"train": b.train, "val": b.val, "holdout": b.holdout}
    for name, df in splits.items():
        try:
            schema.validate(df[b.features] if all(c in df for c in b.features) else df, lazy=True)
            r.add(Check(f"schema[{name}]", PASS, detail=f"{len(df)} rows valid"))
        except Exception as exc:  # pandera SchemaErrors
            cases = getattr(exc, "failure_cases", None)
            n = len(cases) if cases is not None else "?"
            r.add(Check(f"schema[{name}]", FAIL, value=str(n), threshold="0", detail=f"schema violations: {str(exc)[:200]}"))
    floor = cfg.get("min_rows", {"train": 1000, "val": 200, "holdout": 200})
    for name, df in splits.items():
        mn = floor.get(name, 0)
        r.add(Check(f"row_floor[{name}]", PASS if len(df) >= mn else FAIL, value=len(df), threshold=mn))
    max_null = cfg.get("max_null_rate", 0.0)
    worst = max((float(df[b.features].isna().mean().max()) for df in splits.values()), default=0.0)
    r.add(Check("null_rate", PASS if worst <= max_null else FAIL, value=round(worst, 6), threshold=max_null))
    if b.id_col:
        ids = {n: set(df[b.id_col]) for n, df in splits.items()}
        if b.stream is not None:
            ids["stream"] = set(b.stream[b.id_col])
        names = list(ids)
        overlaps = {f"{a}&{c}": len(ids[a] & ids[c]) for i, a in enumerate(names) for c in names[i + 1:]}
        bad = {k: v for k, v in overlaps.items() if v}
        r.add(Check("split_leakage", FAIL if bad else PASS, value=str(bad) if bad else "0 shared ids", threshold="0",
                    detail="no id may appear in more than one split"))
    else:
        r.add(Check("split_leakage", WARN, detail="project declares no id_col; leakage not checked"))
    return r


def g2_metric_floors(ctx: GateContext) -> GateResult:
    r = GateResult("G2", "Metric floors")
    floors = ctx.cfg.get("g2_floors", {})
    for m, rule in floors.items():
        v = ctx.eval.metrics.get(m)
        if v is None:
            r.add(Check(m, FAIL, detail="metric not produced by project.metrics()"))
            continue
        lo, hi = rule.get("min"), rule.get("max")
        ok = (lo is None or v >= lo) and (hi is None or v <= hi)
        r.add(Check(m, PASS if ok else FAIL, value=round(v, 6), threshold=f"min {lo}" if lo is not None else f"max {hi}",
                    ci=ctx.eval.metric_ci.get(m)))
    return r


def g3_no_regression(ctx: GateContext) -> GateResult:
    r = GateResult("G3", "No regression vs champion")
    if ctx.champion is None or ctx.champion_eval is None:
        r.status, r.note = SKIP, "No champion exists: first promotion. G3 skipped and recorded as skipped."
        r.add(Check("regression", SKIP, detail=r.note))
        return r
    tol = ctx.cfg.get("g3_regression", {})
    dirs = ctx.project.metric_directions()
    ce = ctx.champion_eval
    if len(ce.y) != len(ctx.eval.y) or not np.array_equal(ce.y, ctx.eval.y):
        r.add(Check("same_holdout", FAIL, detail="champion and candidate were scored on different holdouts"))
        return r
    diffs = paired_bootstrap(ctx.project.score, ctx.eval.y, ctx.eval.pred, ce.pred, ctx.eval.w, ctx.n_boot, ctx.seed)
    for m, rule in tol.get("metrics", {}).items():
        sign = 1.0 if dirs[m] == "lower" else -1.0   # positive worseness = worse
        tol_rel = rule.get("max_relative_worsening", 0.0)
        base = abs(ce.metrics[m]) or 1.0
        worse = sign * (ctx.eval.metrics[m] - ce.metrics[m]) / base
        w_ci = [sign * x / base for x in _ci(diffs[m])]
        w_hi = max(w_ci)
        status = FAIL if worse > tol_rel else (WARN if w_hi > tol_rel else PASS)
        r.add(Check(m, status, value=round(ctx.eval.metrics[m], 6), champion_value=round(ce.metrics[m], 6),
                    threshold=f"relative worsening <= {tol_rel}", ci=[round(min(w_ci), 6), round(w_hi, 6)],
                    detail=f"relative worsening {worse:+.4%} (95% CI of worsening shown)"))
    return r


def g4_calibration(ctx: GateContext) -> GateResult:
    r = GateResult("G4", "Calibration")
    for key, rule in ctx.cfg.get("g4_calibration", {}).items():
        v = ctx.eval.calibration.get(key)
        if v is None:
            r.add(Check(key, FAIL, detail="not produced by project.calibration_report()"))
            continue
        r.add(Check(key, PASS if v <= rule["max"] else FAIL, value=round(v, 6), threshold=f"max {rule['max']}"))
    return r


def g5_slices(ctx: GateContext) -> GateResult:
    r = GateResult("G5", "Slice performance")
    cfg = ctx.cfg.get("g5_slices", {})
    metric, max_gap = cfg.get("metric"), cfg.get("max_gap", 0.2)
    target = cfg.get("target", "overall")
    overall = ctx.eval.metrics.get(metric, ctx.project.score(ctx.eval.y, ctx.eval.pred, ctx.eval.w).get(metric))
    ref = overall if target == "overall" else float(target)
    for sname, rows in ctx.eval.slices.items():
        for row in rows:
            name = f"{sname}={row['level']}"
            if row["n"] < row["min_samples"]:
                r.add(Check(name, WARN, value="insufficient data", threshold=f"n >= {row['min_samples']}",
                            detail=f"n={row['n']}: slice too small to judge; reported, not passed silently"))
                continue
            v = row["scores"][metric]
            gap = abs(v - ref) / (abs(ref) or 1.0)
            lo, hi = row["ci"][metric]
            gap_hi = max(abs(lo - ref), abs(hi - ref)) / (abs(ref) or 1.0)
            status = FAIL if gap > max_gap else (WARN if gap_hi > max_gap else PASS)
            r.add(Check(name, status, value=round(v, 6), threshold=f"|gap| <= {max_gap} vs {round(ref, 4)}",
                        ci=[round(lo, 6), round(hi, 6)], detail=f"n={row['n']}, gap {gap:.1%}"))
    if not r.checks:
        r.add(Check("slices", WARN, detail="project declares no slices"))
    return r
