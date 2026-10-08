"""`make demo`: the whole release cycle, offline, tagged synthetic where simulated.

Story: train -> gate -> bootstrap champion (GLM) -> degraded challenger rejected -> good challenger (GBM) through
shadow + canary -> promoted -> synthetic drift ramp -> ok/watch/alert -> rollback recommendation -> rollback ->
audit chain verified -> evidence saved -> report viewer built.
"""

from __future__ import annotations

import json
import shutil
import time

from . import promote as P
from .incident import build_incident
from .monitoring.drift import monitor_windows, recommend
from .monitoring.injection import generic_scenarios
from .monitoring.matrix import run_matrix
from .provenance import load_yaml
from .rollback import rollback
from .serving.app import attach_test_client
from .traffic import run_traffic
from .workspace import Workspace

DEFAULTS = {
    "families": {"baseline": "constant", "champion": "glm", "challenger": "gbm", "degraded": "gbm_degraded"},
    "scenario": "young_driver_surge", "baseline_windows": 2, "ramp": [0.0, 0.04, 0.09, 0.2, 0.5, 1.0], "after_windows": 2, "per_window": 2500,
}


class Step:
    def __init__(self, title: str):
        self.title, self.t0 = title, time.perf_counter()

    def __enter__(self):
        print(f"\n== {self.title}", flush=True)
        return self

    def __exit__(self, *exc):
        print(f"   ({time.perf_counter() - self.t0:.1f}s)", flush=True)


def _reset(ws: Workspace) -> None:
    for p in (ws.registry.root, ws.reports_dir, ws.logs_dir, ws.audit.path.parent):
        if p.exists():
            shutil.rmtree(p)


def run_demo(ws: Workspace, fast: bool = False) -> int:
    t_start = time.perf_counter()
    cfg = {**DEFAULTS, **(load_yaml(ws.root / "config" / "project.yaml").get("demo") or {})}
    fam = cfg["families"]
    per_window = 1200 if fast else int(cfg["per_window"])
    if fast and hasattr(ws.project, "options"):
        ws.project.options.update({"data_source": "synthetic", "sample": 40000})
    syn_gates = ws.root / "config" / "gates.synthetic.yaml"
    if fast and syn_gates.exists():
        ws.gates_path = syn_gates   # smoke thresholds for the small synthetic stand-in; see the file header
        print("   (fast mode: synthetic stand-in data + config/gates.synthetic.yaml; proves the machinery, not model quality)")
    _reset(ws)
    with Step("1/11 setup check"):
        b = ws.bundle()
        print(f"   data {b.data_version} | train {len(b.train)} val {len(b.val)} holdout {len(b.holdout)} stream {len(b.stream)}"
              f"{' | SYNTHETIC STAND-IN DATA' if b.meta.get('synthetic') else ''}")
        print(f"   split: {b.meta.get('split', 'n/a')}")
    hashes: dict[str, str] = {}
    with Step("2/11 train constant, champion and challenger families (CV tuning inside train only)"):
        for role in ("baseline", "champion", "challenger"):
            params = {} if (fam[role] == "constant" or fast or not hasattr(ws.project, "tune")) else ws.project.tune(
                b, fam[role], ws.seed, ws.reports_dir / "tuning")
            c = P.train_candidate(ws, fam[role], params)
            hashes[role] = c.hash
            print(f"   {role:<10} {fam[role]:<9} {c.hash}  (train {c.notes['train_seconds']}s)")
    with Step("3/11 gate + bootstrap the first champion (no champion yet: G3 is skipped and recorded)"):
        rep = P.gate_candidate(ws, ws.registry.load(hashes["champion"]))
        print(f"   gates: {rep['overall']} {[(g['id'], g['status']) for g in rep['gates']]}")
        assert rep["overall"] == "pass", rep["failed_gates"]
        P.register_candidate(ws, hashes["champion"])
        P.promote(ws, hashes["champion"], actor="demo")
    client = attach_test_client(ws)
    with Step("4/11 degraded challenger must be rejected"):
        bad = P.train_candidate(ws, fam["degraded"], None)
        r = P.gate_candidate(ws, bad)
        print(f"   {bad.hash}: {r['overall']} (failed gates: {r['failed_gates']})")
        assert r["overall"] == "fail", "degraded challenger should have failed a gate"
        try:
            P.register_candidate(ws, bad.hash)
            raise AssertionError("promotion path must refuse a failing candidate")
        except P.PromotionRefused as exc:
            print(f"   registration refused: {exc}")
    cmp_fn = None
    try:  # three-model comparison table (claims example); optional for other projects
        from examples.claims_frequency.compare import compare as cmp_fn  # noqa: PLC0415
    except Exception:
        pass
    ch = hashes["challenger"]
    with Step("5/11 challenger through the gates (G3 compares against the champion, same holdout, paired bootstrap)"):
        if cmp_fn:
            cmp_fn(ws, {"Constant baseline": hashes["baseline"], "Poisson GLM": hashes["champion"], "LightGBM": ch})
        rep = P.gate_candidate(ws, ws.registry.load(ch))
        g3 = next(g for g in rep["gates"] if g["id"] == "G3")
        print(f"   gates: {rep['overall']} {[(g['id'], g['status']) for g in rep['gates']]}")
        for c in g3["checks"]:
            print(f"   G3 {c['name']}: {c['detail']}")
        assert rep["overall"] == "pass", rep["failed_gates"]
    win = 0
    with Step("6/11 shadow (simulated traffic; candidate scored but never returned)"):
        P.register_candidate(ws, ch)
        P.start_shadow(ws, ch)
        n_sh = 2 if not fast else 3
        run_traffic(ws, client, n_windows=n_sh, per_window=per_window if not fast else 1200, seed=11, start_window=win)
        win += n_sh
        sh = P.evaluate_shadow(ws, ch)
        print(f"   shadow {sh['verdict']}: requests {sh['requests']}, spearman {sh['spearman']:.3f}, agreement {sh['agreement_rate']:.3f}")
        assert sh["verdict"] == "pass", sh["checks"]
    with Step("7/11 canary 10% (stable-hash routing) then promote"):
        P.start_canary(ws, ch)
        n_ca = 2 if not fast else 3
        run_traffic(ws, client, n_windows=n_ca, per_window=per_window if not fast else 1200, seed=12, start_window=win)
        win += n_ca
        ca = P.evaluate_canary(ws, ch)
        print(f"   canary {ca['verdict']}: {ca['requests_canary']} canary / {ca['requests_control']} control requests")
        assert ca["verdict"] == "pass", ca["checks"]
        e = P.promote(ws, ch, actor="demo")
        print(f"   promoted (audit seq {e['seq']}); previous champion kept for rollback")
    scen_name = cfg["scenario"]
    sc = getattr(ws.project, "drift_scenarios", lambda: {})()
    if scen_name not in sc:
        feats = ws.project.monitored_features()
        num = next(f for f in feats if b.reference[f].dtype.kind in "if")
        sc = generic_scenarios(num, num)
        scen_name = "covariate_shift"
    scenario = sc[scen_name]
    ramp = cfg["ramp"] if not fast else [0.0, 0.09, 0.3, 1.0]
    with Step(f"8/11 serve steady traffic, then inject SYNTHETIC drift: {scen_name} (ramp {ramp})"):
        before = list(range(win, win + cfg["baseline_windows"]))
        sent = run_traffic(ws, client, n_windows=len(before), per_window=per_window, seed=21, start_window=win)
        win += len(before)
        during = list(range(win, win + len(ramp)))
        sent += run_traffic(ws, client, n_windows=len(ramp), per_window=per_window, seed=22, scenario=scenario, intensities=ramp,
                            start_window=win, scenario_name=scen_name)
        win += len(ramp)
    with Step("9/11 monitor drift, recommend, roll back"):
        tl = monitor_windows(ws, "incident", windows=before + during)
        print("   levels by window:", {w["window"]: w["level"] for w in tl["windows"]})
        rec = recommend(ws, tl, "incident")
        assert rec is not None, "expected the ramp to reach alert"
        print(f"   audit: {rec['reason']}")
        out = rollback(ws, f"Drift alert from synthetic scenario '{scen_name}'; restoring previous champion per runbook.",
                       evidence=[{"kind": "drift_summary", "path": "drift/incident/timeline.json", "summary": "alert; see window files"}],
                       actor="demo", synthetic=True)
        print(f"   rollback: {out}")
        again = rollback(ws, "idempotency check")
        print(f"   second rollback call is a no-op: {again['status']}")
        after = list(range(win, win + cfg["after_windows"]))
        sent += run_traffic(ws, client, n_windows=len(after), per_window=per_window, seed=23, scenario=scenario, intensities=[ramp[-1]] * len(after),
                            start_window=win, scenario_name=scen_name)
        tl2 = monitor_windows(ws, "incident", windows=before + during + after)
        print("   levels incl. after rollback:", {w["window"]: w["level"] for w in tl2["windows"]})
    with Step("10/11 drift scenario matrix: every declared detection vs what was observed (synthetic, scratch logs)"):
        mx = run_matrix(ws, per_window=1500 if fast else 2500)
        for r in mx["scenarios"]:
            print(f"   {r['scenario']:<26} {'detected' if r['detected'] else 'MISSED  '} {' '.join(r['window_levels'])}")
    with Step("11/11 verify audit chain, save incident evidence, build report viewer"):
        v = ws.audit.verify()
        print(f"   audit chain: {'OK' if v.ok else 'BROKEN: ' + str(v.error)} ({v.entries} entries)")
        assert v.ok
        build_incident(ws, scenario=scen_name, windows={"before": before, "during": during, "after": after}, timeline=tl2,
                       intensities={s["window"]: s["intensity"] for s in sent})
        card = ws.report_dir(ch) / "MODEL_CARD.md"
        docs = ws.root / "examples" / "claims_frequency" / "docs"
        if card.exists() and docs.exists() and not fast:
            shutil.copy(card, docs / "MODEL_CARD.md")
        try:
            from .viewer.build import build_viewer
            path = build_viewer(ws, ws.reports_dir / "viewer" / "index.html")
            print(f"   report viewer: {path}")
        except ImportError:
            print("   (report viewer not available)")
    total = time.perf_counter() - t_start
    print(f"\nDemo finished in {total / 60:.1f} min. Synthetic parts: all traffic, drift and labels-in-traffic. Open the viewer above.")
    (ws.reports_dir / "demo_runtime.json").write_text(json.dumps({"seconds": round(total, 1), "fast": fast}))
    return 0
