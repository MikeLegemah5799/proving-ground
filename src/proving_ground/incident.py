"""Assemble the headline-incident evidence: per-window champion calibration before / during / after."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .promote import read_jsonl
from .workspace import Workspace


def window_calibration(ws: Workspace, win: int, rows: list[dict[str, Any]], labels: dict[str, float]) -> dict[str, Any] | None:
    b = ws.bundle()
    lr = [r for r in rows if not r.get("invalid") and r["request_id"] in labels]
    if len(lr) < 50:
        return None
    y = np.array([labels[r["request_id"]] for r in lr])
    p = np.array([r["champion_pred"] for r in lr])
    w = np.array([r["features"].get(b.weight, 1.0) for r in lr], float) if b.weight else None
    out: dict[str, Any] = {"n": len(lr), "scores": {k: float(v) for k, v in ws.project.score(y, p, w).items()}}
    fn = getattr(ws.project, "calibration_arrays", None)
    if fn:
        out["calibration"] = fn(y, p, w)
    return out


def build_incident(ws: Workspace, *, scenario: str, windows: dict[str, list[int]], timeline: dict[str, Any], intensities: dict[int, float] | None = None, note: str = "") -> dict[str, Any]:
    """windows: {"before": [...], "during": [...], "after": [...]} (window ids)."""
    rows_all = read_jsonl(ws.logs_dir / "requests.jsonl")
    labels = {r["request_id"]: r["label"] for r in read_jsonl(ws.logs_dir / "labels.jsonl")}
    by: dict[int, list[dict[str, Any]]] = {}
    for r in rows_all:
        by.setdefault(int(r["window"]), []).append(r)
    lvl = {w["window"]: w for w in timeline["windows"]}
    phases = []
    for phase, wins in windows.items():
        for w in wins:
            rows = by.get(w, [])
            champs = sorted({r["champion_hash"] for r in rows})
            phases.append({
                "phase": phase, "window": w, "level": lvl.get(w, {}).get("level"), "rows": len(rows), "serving_champion": champs[-1] if champs else None,
                "intensity": (intensities or {}).get(w), "start": lvl.get(w, {}).get("window_start"), "end": lvl.get(w, {}).get("window_end"),
                "calibration": window_calibration(ws, w, rows, labels),
            })
    audit = [{k: e[k] for k in ("seq", "ts", "event", "candidate_hash", "reason", "champion_hash_before", "champion_hash_after")} for e in ws.audit.entries()
             if e["event"] in ("hold", "rollback", "promote", "canary_abort", "reject")]
    out = {"schema_version": "1.0", "kind": "incident", "scenario": scenario, "synthetic": True,
           "note": note or "Drift is injected into simulated traffic; nothing here is real-world behavior.",
           "phases": phases, "audit_events": audit, "chain_ok": ws.audit.verify().ok}
    d = ws.reports_dir / "incident"
    d.mkdir(parents=True, exist_ok=True)
    (d / "incident.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    return out
