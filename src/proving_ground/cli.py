"""proving-ground command line. Every `make` target is a thin wrapper around one of these."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import promote as P
from .rollback import RollbackError, rollback
from .workspace import Workspace


def _ws(args) -> Workspace:
    ws = Workspace.from_config(args.root, args.config)
    return ws


def _client(ws: Workspace, url: str | None):
    if url:
        import httpx

        c = httpx.Client(base_url=url, timeout=60)
        hdr = {"X-Admin-Token": os.environ.get("PG_ADMIN_TOKEN", "demo-token")}
        ws.reload_fn = lambda: c.post("/admin/reload", headers=hdr).raise_for_status()
        ws.health_fn = lambda: c.get("/health").json()
        return c
    from .serving.app import attach_test_client

    return attach_test_client(ws)


def _pointer(ws: Workspace) -> Path:
    return ws.registry.root / "trained.json"


def _remember(ws: Workspace, family: str, h: str) -> None:
    p = _pointer(ws)
    d = json.loads(p.read_text()) if p.exists() else {}
    d[family] = h
    d["last"] = h
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2))


def _resolve(ws: Workspace, args) -> str:
    if getattr(args, "candidate", None):
        return args.candidate
    d = json.loads(_pointer(ws).read_text()) if _pointer(ws).exists() else {}
    fam = getattr(args, "family", None)
    h = d.get(fam) if fam else d.get("last")
    if not h:
        raise SystemExit("no candidate: pass --candidate HASH or train one first")
    return h


def cmd_train(args, ws: Workspace) -> int:
    family = args.family or ws.release_cfg().get("family") or "glm"
    params: dict[str, Any] = json.loads(args.params) if args.params else {}
    if args.tune and hasattr(ws.project, "tune"):
        params = {**ws.project.tune(ws.bundle(), family, ws.seed, ws.reports_dir / "tuning"), **params}
    cand = P.train_candidate(ws, family, params)
    _remember(ws, family, cand.hash)
    print(json.dumps({"candidate_hash": cand.hash, "family": family, "train_seconds": cand.notes.get("train_seconds")}))
    return 0


def cmd_gate(args, ws: Workspace) -> int:
    h = _resolve(ws, args)
    rep = P.gate_candidate(ws, ws.registry.load(h))
    print(f"gate report {rep['overall'].upper()} for {h}: {ws.report_dir(h) / 'gate_report.md'}")
    for g in rep["gates"]:
        print(f"  {g['id']} {g['name']:<28} {g['status']}")
    return 0 if rep["overall"] == "pass" else 1


def cmd_promote_shadow(args, ws: Workspace) -> int:
    h = _resolve(ws, args)
    P.register_candidate(ws, h)
    P.start_shadow(ws, h)
    print(f"{h} registered; stage=shadow")
    return 0


def cmd_traffic(args, ws: Workspace) -> int:
    from .traffic import run_traffic

    client = _client(ws, args.url)
    s = run_traffic(ws, client, n_windows=args.windows, per_window=args.per_window, seed=args.seed, start_window=args.start_window)
    print(json.dumps(s))
    return 0


def cmd_shadow_eval(args, ws: Workspace) -> int:
    s = P.evaluate_shadow(ws, _resolve(ws, args))
    print(f"shadow verdict: {s['verdict']} ({s['requests']} requests)")
    return 0 if s["verdict"] == "pass" else (2 if s["verdict"] == "extend" else 1)


def cmd_promote_canary(args, ws: Workspace) -> int:
    P.start_canary(ws, _resolve(ws, args))
    print("stage=canary")
    return 0


def cmd_canary_eval(args, ws: Workspace) -> int:
    s = P.evaluate_canary(ws, _resolve(ws, args))
    print(f"canary verdict: {s['verdict']} ({s['requests_canary']} canary requests)")
    return 0 if s["verdict"] == "pass" else (2 if s["verdict"] == "extend" else 1)


def cmd_promote(args, ws: Workspace) -> int:
    e = P.promote(ws, _resolve(ws, args))
    print(f"promoted; audit seq {e['seq']}")
    return 0


def cmd_rollback(args, ws: Workspace) -> int:
    client = _client(ws, args.url) if args.url else None  # noqa: F841 (installs reload/health hooks)
    try:
        out = rollback(ws, args.reason, actor=args.actor)
    except RollbackError as exc:
        print(f"rollback failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(out))
    return 0


def cmd_audit_verify(args, ws: Workspace) -> int:
    r = ws.audit.verify()
    print(f"audit chain OK: {r.entries} entries, head {r.head[:16]}" if r.ok else f"AUDIT CHAIN BROKEN: {r.error}")
    return 0 if r.ok else 1


def cmd_serve(args, ws: Workspace) -> int:
    import uvicorn

    from .serving.app import create_app

    uvicorn.run(create_app(ws), host=args.host, port=args.port)
    return 0


def cmd_inject_drift(args, ws: Workspace) -> int:
    from .monitoring.injection import generic_scenarios
    from .traffic import run_traffic

    sc = getattr(ws.project, "drift_scenarios", lambda: {})()
    if args.scenario not in sc:
        feats = ws.project.monitored_features()
        b = ws.bundle()
        num = next(f for f in feats if f in b.reference and b.reference[f].dtype.kind in "if")
        cat = next((f for f in feats if f in b.reference and b.reference[f].dtype.kind not in "if"), num)
        sc = {**generic_scenarios(num, cat, b.weight), **sc}
    if args.scenario not in sc:
        raise SystemExit(f"unknown scenario {args.scenario}; choose from {sorted(sc)}")
    ramp = [float(x) for x in args.ramp.split(",")]
    client = _client(ws, args.url)
    s = run_traffic(ws, client, n_windows=len(ramp), per_window=args.per_window, seed=args.seed, scenario=sc[args.scenario],
                    intensities=ramp, start_window=args.start_window, scenario_name=args.scenario)
    print(json.dumps(s))
    return 0


def cmd_monitor(args, ws: Workspace) -> int:
    from .monitoring.drift import monitor_windows, recommend

    tl = monitor_windows(ws, args.run, html=not args.no_html)
    print("window levels:", {w["window"]: w["level"] for w in tl["windows"]})
    if recommend(ws, tl, args.run):
        print("ALERT: rollback recommended (hold entry written). Run `make rollback`.")
    return 0


def cmd_viewer(args, ws: Workspace) -> int:
    from .viewer.build import build_viewer

    out = build_viewer(ws, Path(args.out) if args.out else ws.reports_dir / "viewer" / "index.html")
    print(f"report viewer: {out}")
    return 0


def cmd_demo(args, ws: Workspace) -> int:
    from .demo import run_demo

    return run_demo(ws, fast=args.fast)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="proving-ground", description="Proving Ground: a gated model release template.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *a, **kw):
        p = sub.add_parser(name, **kw)
        p.set_defaults(fn=fn)
        return p

    p = add("train", cmd_train)
    p.add_argument("--family")
    p.add_argument("--params")
    p.add_argument("--tune", action="store_true")
    for name, fn in (("gate", cmd_gate), ("promote-shadow", cmd_promote_shadow), ("shadow-eval", cmd_shadow_eval),
                     ("promote-canary", cmd_promote_canary), ("canary-eval", cmd_canary_eval), ("promote", cmd_promote)):
        p = add(name, fn)
        p.add_argument("--candidate")
        p.add_argument("--family")
    p = add("traffic", cmd_traffic)
    p.add_argument("--windows", type=int, default=1)
    p.add_argument("--per-window", type=int, default=1000)
    p.add_argument("--start-window", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--url")
    p = add("rollback", cmd_rollback)
    p.add_argument("--reason", required=True)
    p.add_argument("--actor", default="operator")
    p.add_argument("--url")
    add("audit-verify", cmd_audit_verify)
    p = add("serve", cmd_serve)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = add("inject-drift", cmd_inject_drift)
    p.add_argument("--scenario", required=True)
    p.add_argument("--ramp", default="0.4,1.0,1.0")
    p.add_argument("--per-window", type=int, default=1500)
    p.add_argument("--start-window", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--url")
    p = add("monitor", cmd_monitor)
    p.add_argument("--run", default="latest")
    p.add_argument("--no-html", action="store_true")
    p = add("viewer", cmd_viewer)
    p.add_argument("--out")
    p = add("demo", cmd_demo)
    p.add_argument("--fast", action="store_true", help="shortened run for CI smoke tests")
    args = ap.parse_args(argv)
    ws = _ws(args)
    return args.fn(args, ws)


if __name__ == "__main__":
    raise SystemExit(main())
