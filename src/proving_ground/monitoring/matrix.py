"""Run every drift scenario against the current champion and compare with its declared expectation.

Runs in a scratch directory so the real request log and audit log are untouched. Misses are recorded,
never hidden: ``detected: false`` plus what was actually observed.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any

from ..serving.app import attach_test_client
from ..traffic import run_traffic
from ..workspace import Workspace
from .drift import ORDER, monitor_windows
from .injection import Scenario, generic_scenarios


def scenarios_for(ws: Workspace) -> dict[str, Scenario]:
    b = ws.bundle()
    feats = ws.project.monitored_features()
    num = next(f for f in feats if b.reference[f].dtype.kind in "if")
    cat = next((f for f in feats if b.reference[f].dtype.kind not in "if"), num)
    return {**generic_scenarios(num, cat, b.weight), **getattr(ws.project, "drift_scenarios", lambda: {})()}


def _signal_level(win: dict[str, Any], summary: dict[str, Any], signal: str) -> str:
    if signal.startswith("feature:"):
        return summary["features"].get(signal.split(":", 1)[1], {}).get("level", "ok")
    if signal == "validation":
        return summary["validation"]["level"]
    if signal == "label":
        return (summary["label_drift"] or {}).get("level", "ok")
    if signal == "prediction":
        return summary["prediction"]["level"]
    return "ok"


def evaluate_scenario(ws: Workspace, name: str, sc: Scenario, per_window: int = 2500, seed: int = 5) -> dict[str, Any]:
    ws.bundle()
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        w2 = dataclasses.replace(ws, logs_dir=t / "logs", reports_dir=t / "reports", audit=type(ws.audit)(t / "audit.jsonl"))
        client = attach_test_client(w2)
        run_traffic(w2, client, n_windows=1, per_window=per_window, seed=seed, start_window=0)
        ramp = sc.ramp
        run_traffic(w2, client, n_windows=len(ramp), per_window=per_window, seed=seed + 1, scenario=sc, intensities=ramp, start_window=1,
                    scenario_name=name)
        tl = monitor_windows(w2, "matrix", html=False)
        sums = [json.loads((t / "reports" / "drift" / "matrix" / w["file"]).read_text()) for w in tl["windows"]]
    exp = sc.expected
    levels = [s["level"] for s in sums]
    sig_levels = [_signal_level(w, s, exp["signal"]) for w, s in zip(tl["windows"], sums, strict=True)]
    reached = max(levels, key=lambda v: ORDER[v])
    hit = ORDER[max(sig_levels, key=lambda v: ORDER[v])] >= ORDER[exp["level"]]
    first = lambda lv: next((i for i, v in enumerate(levels) if v == lv), None)  # noqa: E731
    watch_first = (first("watch") is not None and first("alert") is not None and first("watch") < first("alert"))
    ok = hit and ORDER[reached] >= ORDER[exp["level"]] and (not exp.get("watch_before_alert") or watch_first)
    return {"scenario": name, "description": sc.description, "synthetic": True, "expected": exp, "ramp": ramp,
            "window_levels": levels, "signal_levels": sig_levels, "max_level": reached, "signal_hit": bool(hit),
            "watch_before_alert": bool(watch_first), "detected": bool(ok),
            "note": "" if ok else "MISS: declared expectation was not met; see window_levels/signal_levels."}


def run_matrix(ws: Workspace, names: list[str] | None = None, per_window: int = 2500) -> dict[str, Any]:
    scs = scenarios_for(ws)
    rows = [evaluate_scenario(ws, n, s, per_window) for n, s in scs.items() if names is None or n in names]
    out = {"schema_version": "1.0", "kind": "scenario_matrix", "synthetic": True, "champion_hash": ws.registry.get_alias("champion"),
           "scenarios": rows, "missed": [r["scenario"] for r in rows if not r["detected"]]}
    d = ws.reports_dir / "drift"
    d.mkdir(parents=True, exist_ok=True)
    (d / "scenario_matrix.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    return out
