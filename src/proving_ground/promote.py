"""Release stages: register -> shadow -> canary -> promote. Every transition is guarded.

No code path promotes a candidate without a passing gate report for that exact candidate
hash, data version and *current* gate config. There is deliberately no force option.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .candidate import Candidate, build_candidate
from .evaluation import evaluate_holdout
from .gates import GateContext, load_gate_report, run_gates
from .monitoring.stats import psi_numeric
from .provenance import utc_now
from .workspace import Workspace

SCHEMA_VERSION = "1.0"


class PromotionRefused(Exception):
    pass


# --------------------------------------------------------------------------- train + gate
def train_candidate(ws: Workspace, family: str, params: dict | None = None, seed: int | None = None) -> Candidate:
    cand = build_candidate(ws.project, ws.bundle(), family=family, seed=ws.seed if seed is None else seed, params=params)
    ws.registry.save(cand, {"train_seconds": cand.notes.get("train_seconds")})
    return cand


def gate_candidate(ws: Workspace, cand: Candidate) -> dict[str, Any]:
    """Run G1..G8. A failing report is audited as a ``reject`` event."""
    cfg = ws.gates_cfg()
    n_boot = int(cfg.get("n_boot", 200))
    rdir = ws.report_dir(cand.hash)
    ledger = ws.reports_dir / "holdout_ledger.jsonl"
    ev = evaluate_holdout(ws.project, ws.bundle(), cand, rdir, ledger, n_boot, ws.seed)
    champ_hash = ws.registry.get_alias("champion")
    champ = champ_ev = None
    if champ_hash and champ_hash != cand.hash:
        champ = ws.registry.load(champ_hash)
        champ_ev = evaluate_holdout(ws.project, ws.bundle(), champ, ws.report_dir(champ_hash), ledger, n_boot, ws.seed)
    ctx = GateContext(ws.project, ws.bundle(), cand, ev, cfg, rdir, champ, champ_ev, ws.seed, n_boot,
                      retrain=lambda: build_candidate(ws.project, ws.bundle(), family=cand.family, seed=cand.seed,
                                                      params=cand.params))
    report = run_gates(ctx, ws.prov(), ws.gate_config_hash())
    from .tracking import log_gate_run

    log_gate_run(ws, report)
    if report["overall"] == "fail":
        ws.audit.append(
            "reject", candidate_hash=cand.hash, champion_hash_before=champ_hash, champion_hash_after=champ_hash,
            git_sha=report["provenance"]["git_sha"], data_version=report["provenance"]["data_version"],
            gate_config_hash=report["gate_config_hash"], synthetic=bool(report["synthetic"]),
            reason=f"Gate(s) {', '.join(report['failed_gates'])} failed; candidate rejected.",
            evidence=[{"kind": "gate_report", "path": f"{cand.hash}/gate_report.json", "summary": "failed: " + ", ".join(
                f"{g['id']}:{c['name']}" for g in report["gates"] for c in g["checks"] if c["status"] == "fail")[:300]}])
    return report


# --------------------------------------------------------------------------- guards
def assert_gate_pass(ws: Workspace, h: str) -> dict[str, Any]:
    rep = load_gate_report(ws.report_dir(h))
    if rep is None:
        raise PromotionRefused(f"no gate report for candidate {h}; run the gates first")
    if rep["candidate_hash"] != h:
        raise PromotionRefused("gate report is for a different candidate hash")
    if rep["overall"] != "pass":
        raise PromotionRefused(f"gate report for {h} did not pass (failed: {rep['failed_gates']})")
    if not ws.registry.exists(h):
        raise PromotionRefused(f"candidate {h} is not in the registry")
    meta = ws.registry.meta(h)
    if rep["provenance"]["data_version"] != meta["data_version"]:
        raise PromotionRefused("gate report data version does not match the candidate's data version")
    if rep["gate_config_hash"] != ws.gate_config_hash():
        raise PromotionRefused("gates.yaml changed since this report was produced; re-run the gates")
    return rep


def register_candidate(ws: Workspace, h: str) -> dict[str, Any]:
    """`make promote-shadow`: set the candidate alias only if the gate report passes."""
    assert_gate_pass(ws, h)
    ws.registry.set_aliases(candidate=h)
    st = ws.registry.set_state(stage="registered", canary_pct=0, rolled_back_from=None)
    ws.reload()
    return st


def start_shadow(ws: Workspace, h: str) -> None:
    _require(ws, h, {"registered", "shadow"})
    ws.registry.set_state(stage="shadow", canary_pct=0)
    ws.reload()


def start_canary(ws: Workspace, h: str) -> None:
    _require(ws, h, {"shadow_passed", "canary"})
    ws.registry.set_state(stage="canary", canary_pct=int(ws.release_cfg()["canary"]["pct"]))
    ws.reload()


def _require(ws: Workspace, h: str, stages: set[str]) -> None:
    assert_gate_pass(ws, h)
    if ws.registry.get_alias("candidate") != h:
        raise PromotionRefused(f"{h} is not the registered candidate")
    st = ws.registry.state()["stage"]
    if st not in stages:
        raise PromotionRefused(f"stage is {st!r}; need one of {sorted(stages)}")


# --------------------------------------------------------------------------- log readers
def read_jsonl(path) -> list[dict[str, Any]]:
    try:
        return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    except FileNotFoundError:
        return []


def _labels(ws: Workspace) -> dict[str, float]:
    return {r["request_id"]: r["label"] for r in read_jsonl(ws.logs_dir / "labels.jsonl")}


def _weight(ws: Workspace, row: dict) -> float:
    wc = ws.bundle().weight
    return float(row["features"].get(wc, 1.0)) if wc else 1.0


def _compare_error(ws: Workspace, rows: list[dict], metric: str, preds_a: str, preds_b: str):
    labels = _labels(ws)
    rr = [r for r in rows if r["request_id"] in labels and r.get(preds_a) is not None and r.get(preds_b) is not None]
    if len(rr) < 30:
        return None
    y = np.array([labels[r["request_id"]] for r in rr], dtype=float)
    w = np.array([_weight(ws, r) for r in rr])
    w = w if ws.bundle().weight else None
    a = ws.project.score(y, np.array([r[preds_a] for r in rr]), w)[metric]
    b = ws.project.score(y, np.array([r[preds_b] for r in rr]), w)[metric]
    return {"n": len(rr), "candidate": float(a), "reference": float(b)}


def _worse_rel(ws: Workspace, metric: str, cand: float, ref: float) -> float:
    lower = ws.project.metric_directions()[metric] == "lower"
    return ((cand - ref) if lower else (ref - cand)) / (abs(ref) or 1.0)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(spearmanr(a, b).statistic)


# --------------------------------------------------------------------------- shadow
def evaluate_shadow(ws: Workspace, h: str) -> dict[str, Any]:
    cfg = ws.release_cfg()["shadow"]
    champ = ws.registry.get_alias("champion")
    rows = [r for r in read_jsonl(ws.logs_dir / "requests.jsonl")
            if r.get("stage") == "shadow" and r.get("candidate_hash") == h and not r.get("invalid")]
    n = len(rows)
    both = [r for r in rows if r.get("candidate_pred") is not None]
    exc = sum(1 for r in rows if r.get("candidate_error"))
    cp = np.array([r["champion_pred"] for r in both]) if both else np.array([])
    dp = np.array([r["candidate_pred"] for r in both]) if both else np.array([])
    agree = float(np.mean(np.abs(dp - cp) <= cfg["agree_rel_tol"] * np.maximum(np.abs(cp), 1e-12))) if both else None
    rho = _spearman(cp, dp) if both else None
    lat = float(np.mean([r["candidate_ms"] - r["champion_ms"] for r in both])) if both else None
    err = _compare_error(ws, both, cfg["compare_metric"], "candidate_pred", "champion_pred")
    checks = []

    def chk(name, value, thr, ok):
        checks.append({"name": name, "value": value, "threshold": thr, "status": "pass" if ok else "fail"})

    exc_rate = exc / n if n else None
    verdict = "extend"
    if n >= cfg["min_requests"]:
        chk("spearman", rho, f">= {cfg['min_spearman']}", rho >= cfg["min_spearman"])
        chk("exception_rate", exc_rate, f"<= {cfg['max_exception_rate']}", exc_rate <= cfg["max_exception_rate"])
        chk("latency_delta_ms", lat, f"<= {cfg['max_latency_delta_ms']}", lat <= cfg["max_latency_delta_ms"])
        if err:
            worse = _worse_rel(ws, cfg["compare_metric"], err["candidate"], err["reference"])
            chk(f"{cfg['compare_metric']}_vs_champion", worse, f"relative worsening <= {cfg['max_relative_error_increase']}",
                worse <= cfg["max_relative_error_increase"])
        verdict = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    summary = {
        "schema_version": SCHEMA_VERSION, "kind": "shadow_summary", "candidate_hash": h, "champion_hash": champ,
        "created": utc_now(), "verdict": verdict, "requests": n, "min_requests": cfg["min_requests"],
        "agreement_rate": agree, "spearman": rho, "mean_latency_delta_ms": lat, "exception_rate": exc_rate,
        "error_vs_champion": err, "compare_metric": cfg["compare_metric"], "checks": checks,
        "synthetic": True, "note": "Traffic is simulated by replaying held-out policies.",
    }
    _write(ws, h, "shadow_summary.json", summary)
    if verdict == "pass":
        ws.registry.set_state(stage="shadow_passed")
    return summary


# --------------------------------------------------------------------------- canary
def evaluate_canary(ws: Workspace, h: str) -> dict[str, Any]:
    """Check guardrails; a breach aborts the canary to 0% and writes an audit entry."""
    cfg = ws.release_cfg()["canary"]
    champ = ws.registry.get_alias("champion")
    allrows = read_jsonl(ws.logs_dir / "requests.jsonl")
    rows = [r for r in allrows if r.get("stage") == "canary" and r.get("candidate_hash") == h]
    valid = [r for r in rows if not r.get("invalid")]
    routed = [r for r in valid if r["route"] in ("canary", "canary_error")]
    control = [r for r in valid if r["route"] == "champion"]
    shadow_rows = [r for r in allrows if r.get("stage") == "shadow" and r.get("candidate_hash") == h and r.get("candidate_pred") is not None]
    errors = sum(1 for r in routed if r["route"] == "canary_error")
    err_rate = errors / len(routed) if routed else None
    lat = [r["candidate_ms"] for r in routed if r.get("candidate_ms") is not None]
    p95 = float(np.percentile(lat, 95)) if lat else None
    psi_pred = None
    cps = [r["candidate_pred"] for r in routed if r.get("candidate_pred") is not None]
    if len(cps) >= 30 and len(shadow_rows) >= 30:
        psi_pred = psi_numeric(np.array([r["candidate_pred"] for r in shadow_rows]), np.array(cps),
                               bins=int(max(3, min(10, len(cps) // 30))))  # few canary rows -> fewer bins (PSI is biased upward on small samples)
    err_c = _compare_error(ws, [r for r in routed if r["route"] == "canary"], cfg["compare_metric"], "candidate_pred", "champion_pred")
    checks: list[dict[str, Any]] = []
    breach = False

    def chk(name, value, thr, ok):
        nonlocal breach
        breach |= not ok
        checks.append({"name": name, "value": value, "threshold": thr, "status": "pass" if ok else "fail"})

    if err_rate is not None:
        chk("error_rate", err_rate, f"<= {cfg['max_error_rate']}", err_rate <= cfg["max_error_rate"])
    if p95 is not None:
        chk("p95_latency_ms", p95, f"<= {cfg['max_p95_latency_ms']}", p95 <= cfg["max_p95_latency_ms"])
    if psi_pred is not None:
        chk("prediction_psi_vs_shadow", psi_pred, f"<= {cfg['max_pred_psi']}", psi_pred <= cfg["max_pred_psi"])
    if err_c:
        worse = _worse_rel(ws, cfg["compare_metric"], err_c["candidate"], err_c["reference"])
        chk(f"{cfg['compare_metric']}_vs_control", worse, f"relative worsening <= {cfg['max_relative_error_increase']}",
            worse <= cfg["max_relative_error_increase"])
    verdict = "fail" if breach else ("pass" if len(routed) >= cfg["min_requests"] else "extend")
    summary = {
        "schema_version": SCHEMA_VERSION, "kind": "canary_summary", "candidate_hash": h, "champion_hash": champ,
        "created": utc_now(), "verdict": verdict, "canary_pct": cfg["pct"], "requests_total": len(valid),
        "requests_canary": len(routed), "requests_control": len(control), "min_requests": cfg["min_requests"],
        "error_rate": err_rate, "p95_latency_ms": p95, "prediction_psi_vs_shadow": psi_pred,
        "error_vs_control": err_c, "compare_metric": cfg["compare_metric"], "checks": checks,
        "aborted": verdict == "fail", "synthetic": True,
    }
    _write(ws, h, "canary_summary.json", summary)
    if verdict == "pass":
        ws.registry.set_state(stage="canary_passed")
    elif verdict == "fail":
        ws.registry.set_state(stage="aborted", canary_pct=0)
        ws.registry.set_aliases(candidate=None)
        ws.reload()
        failing = [c["name"] for c in checks if c["status"] == "fail"]
        ws.audit.append("canary_abort", candidate_hash=h, champion_hash_before=champ, champion_hash_after=champ,
                        git_sha=ws.prov()["git_sha"], data_version=ws.bundle().data_version, gate_config_hash=ws.gate_config_hash(),
                        synthetic=True, reason=f"Canary guardrail breached ({', '.join(failing)}); traffic returned to 0%.",
                        evidence=[{"kind": "canary_summary", "path": f"{h}/canary_summary.json", "summary": "; ".join(
                            f"{c['name']}={c['value']}" for c in checks if c["status"] == "fail")[:300]}])
    return summary


def _write(ws: Workspace, h: str, name: str, obj: dict) -> None:
    d = ws.report_dir(h)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _load(ws: Workspace, h: str, name: str) -> dict | None:
    p = ws.report_dir(h) / name
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- promote
def promote(ws: Workspace, h: str, actor: str = "proving-ground") -> dict[str, Any]:
    rep = assert_gate_pass(ws, h)
    if ws.registry.get_alias("candidate") != h:
        raise PromotionRefused(f"{h} is not the registered candidate; run promote-shadow first")
    champ = ws.registry.get_alias("champion")
    stage = ws.registry.state()["stage"]
    evidence = [{"kind": "gate_report", "path": f"{h}/gate_report.json", "summary": "overall pass"}]
    if champ is None:
        if stage != "registered":
            raise PromotionRefused(f"stage is {stage!r}; bootstrap promotion needs 'registered'")
        reason = "Bootstrap promotion: no champion existed, so G3 was skipped and shadow/canary had nothing to compare against."
    else:
        if stage != "canary_passed":
            raise PromotionRefused(f"stage is {stage!r}; promotion requires shadow and canary to pass ('canary_passed')")
        sh, ca = _load(ws, h, "shadow_summary.json"), _load(ws, h, "canary_summary.json")
        if not sh or sh["verdict"] != "pass" or not ca or ca["verdict"] != "pass":
            raise PromotionRefused("shadow/canary summaries missing or not passing for this candidate")
        evidence += [{"kind": "shadow_summary", "path": f"{h}/shadow_summary.json", "summary": "verdict pass"},
                     {"kind": "canary_summary", "path": f"{h}/canary_summary.json", "summary": "verdict pass"}]
        reason = "Passed gates, shadow and canary; promoted to champion."
    ws.registry.set_aliases(champion=h, previous_champion=champ, candidate=None)
    ws.registry.set_state(stage="stable", canary_pct=0, rolled_back_from=None)
    ws.reload()
    return ws.audit.append("promote", candidate_hash=h, champion_hash_before=champ, champion_hash_after=h,
                           git_sha=rep["provenance"]["git_sha"], data_version=rep["provenance"]["data_version"],
                           gate_config_hash=rep["gate_config_hash"], evidence=evidence, reason=reason, actor=actor,
                           synthetic=bool(rep.get("synthetic")))
