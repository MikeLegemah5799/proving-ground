from __future__ import annotations

import pickle
import re
import time

import numpy as np

from .base import FAIL, PASS, WARN, Check, GateContext, GateResult


def g6_performance(ctx: GateContext) -> GateResult:
    r = GateResult("G6", "Performance budget")
    cfg, b = ctx.cfg.get("g6_budget", {}), ctx.bundle
    n = cfg.get("latency_samples", 200)
    rows = b.reference[b.features].head(n)
    ctx.project.predict(ctx.candidate.model, rows.head(5))  # warm-up
    times = []
    for i in range(len(rows)):
        t0 = time.perf_counter()
        ctx.project.predict(ctx.candidate.model, rows.iloc[[i]])
        times.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(times, 95))
    mx = cfg.get("p95_latency_ms", 50)
    r.add(Check("p95_latency_ms", PASS if p95 <= mx else FAIL, value=round(p95, 3), threshold=f"max {mx}",
                detail=f"single-row predict(), n={len(times)}"))
    size = len(pickle.dumps(ctx.candidate, protocol=pickle.HIGHEST_PROTOCOL)) / 1e6
    mxs = cfg.get("max_artifact_mb", 100)
    r.add(Check("artifact_mb", PASS if size <= mxs else FAIL, value=round(size, 3), threshold=f"max {mxs}"))
    return r


def g7_reproducibility(ctx: GateContext) -> GateResult:
    r = GateResult("G7", "Reproducibility")
    cfg = ctx.cfg.get("g7_repro", {})
    if ctx.retrain is None:
        r.add(Check("retrain", WARN, detail="no retrain callable supplied; reproducibility not checked"))
        return r
    again = ctx.retrain()
    if again.hash == ctx.candidate.hash:
        r.add(Check("same_hash", PASS, value=again.hash, threshold=ctx.candidate.hash, detail="identical hash on retrain with same seed"))
        return r
    eps = cfg.get("metric_epsilon", 1e-6)
    m2 = ctx.project.metrics(again.model, ctx.bundle)  # reads holdout again only on a hash mismatch; documented
    worst = max((abs(m2[k] - v) / (abs(v) or 1.0) for k, v in ctx.eval.metrics.items() if k in m2), default=0.0)
    r.add(Check("metrics_within_epsilon", PASS if worst <= eps else FAIL, value=float(worst), threshold=f"max {eps}",
                detail=f"hash differs ({again.hash} vs {ctx.candidate.hash}); metrics compared instead"))
    return r


def g8_model_card(ctx: GateContext) -> GateResult:
    r = GateResult("G8", "Model card")
    required = ctx.cfg.get("g8_card", {}).get("required_sections", [])
    text = ctx.card_text
    if not text:
        r.add(Check("card_exists", FAIL, detail="no model card generated for this candidate"))
        return r
    r.add(Check("card_exists", PASS))
    r.add(Check("card_matches_candidate", PASS if ctx.candidate.hash in text else FAIL,
                detail="card must name this candidate hash (regenerated per candidate)"))
    headings = {m.group(1).strip().lower() for m in re.finditer(r"^#{1,4}\s+(.+)$", text, re.M)}
    missing = [s for s in required if s.lower() not in headings]
    r.add(Check("required_sections", FAIL if missing else PASS, value=",".join(missing) or "all present",
                threshold=f"{len(required)} sections", detail=f"missing: {missing}" if missing else ""))
    return r
