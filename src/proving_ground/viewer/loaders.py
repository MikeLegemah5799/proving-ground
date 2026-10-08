"""Read-only loading of everything the viewer renders. Missing inputs yield None / empty, never errors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audit import VerifyResult, read_entries, verify_entries
from .schemas import (
    AuditEntry,
    CanarySummary,
    DriftTimeline,
    DriftWindow,
    GateReport,
    Incident,
    ModelComparison,
    ScenarioMatrix,
    ShadowSummary,
    check_version,
)


@dataclass
class ViewerData:
    reports_dir: Path
    audit: list[AuditEntry] = field(default_factory=list)
    raw_audit: list[dict[str, Any]] = field(default_factory=list)
    chain: VerifyResult | None = None
    gate_reports: list[GateReport] = field(default_factory=list)
    shadow: dict[str, ShadowSummary] = field(default_factory=dict)
    canary: dict[str, CanarySummary] = field(default_factory=dict)
    drift: dict[str, tuple[DriftTimeline, list[DriftWindow]]] = field(default_factory=dict)
    matrix: ScenarioMatrix | None = None
    incident: Incident | None = None
    comparison: ModelComparison | None = None
    cards: dict[str, str] = field(default_factory=dict)       # candidate hash -> markdown
    registry_state: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _json(p: Path, kind: str):
    obj = json.loads(p.read_text(encoding="utf-8"))
    check_version(obj, kind, p.name)
    return obj


def load_viewer_data(reports_dir: Path, audit_path: Path, registry_dir: Path | None = None) -> ViewerData:
    d = ViewerData(reports_dir=reports_dir)
    # --- audit log (verified here, at build time, with the same canonical serialization) -------------------
    try:
        d.raw_audit = read_entries(audit_path)
        d.chain = verify_entries(d.raw_audit)
    except json.JSONDecodeError as exc:
        d.raw_audit, d.chain = [], VerifyResult(False, 0, f"audit log is not valid JSONL: {exc}")
    for e in d.raw_audit:
        try:
            d.audit.append(AuditEntry(**e))
        except Exception as exc:   # a malformed entry is a chain problem, show it
            d.notes.append(f"audit entry seq={e.get('seq')} could not be parsed: {exc}")
    # --- per-candidate reports ------------------------------------------------------------------------------
    if reports_dir.exists():
        for p in sorted(reports_dir.glob("*/gate_report.json")):
            d.gate_reports.append(GateReport(**_json(p, "gate_report")))
        for p in sorted(reports_dir.glob("*/shadow_summary.json")):
            s = ShadowSummary(**_json(p, "shadow_summary"))
            d.shadow[s.candidate_hash] = s
        for p in sorted(reports_dir.glob("*/canary_summary.json")):
            c = CanarySummary(**_json(p, "canary_summary"))
            d.canary[c.candidate_hash] = c
        for p in sorted(reports_dir.glob("*/MODEL_CARD.md")):
            d.cards[p.parent.name] = p.read_text(encoding="utf-8")
        for tl_path in sorted((reports_dir / "drift").glob("*/timeline.json")) if (reports_dir / "drift").exists() else []:
            tl = DriftTimeline(**_json(tl_path, "drift_timeline"))
            wins = []
            for w in tl.windows:
                wp = tl_path.parent / w["file"]
                if wp.exists():
                    wins.append(DriftWindow(**_json(wp, "drift_summary")))
            d.drift[tl.run] = (tl, wins)
        mp = reports_dir / "drift" / "scenario_matrix.json"
        if mp.exists():
            d.matrix = ScenarioMatrix(**_json(mp, "scenario_matrix"))
        ip = reports_dir / "incident" / "incident.json"
        if ip.exists():
            d.incident = Incident(**_json(ip, "incident"))
        cp = reports_dir / "model_comparison.json"
        if cp.exists():
            d.comparison = ModelComparison(**_json(cp, "model_comparison"))
        mf = reports_dir / "demo_runtime.json"
        if mf.exists():
            d.manifest = json.loads(mf.read_text())
    if registry_dir and (registry_dir / "state.json").exists():
        d.registry_state = {**json.loads((registry_dir / "state.json").read_text()),
                            "aliases": json.loads((registry_dir / "aliases.json").read_text()) if (registry_dir / "aliases.json").exists() else {}}
    return d
