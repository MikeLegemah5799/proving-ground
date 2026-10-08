from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..provenance import utc_now
from .base import FAIL, GateContext
from .g1_g5 import g1_data_validation, g2_metric_floors, g3_no_regression, g4_calibration, g5_slices
from .g6_g8 import g6_performance, g7_reproducibility, g8_model_card

SCHEMA_VERSION = "1.0"
REQUIRED_CARD_SECTIONS_DEFAULT = [
    "Intended use", "Out-of-scope use", "Data", "Training procedure", "Metrics", "Slices and calibration",
    "Explainability", "Limitations", "Provenance",
]


def run_gates(ctx: GateContext, prov: dict[str, Any], gate_config_hash: str) -> dict[str, Any]:
    results = [g1_data_validation(ctx), g2_metric_floors(ctx), g3_no_regression(ctx), g4_calibration(ctx),
               g5_slices(ctx), g6_performance(ctx), g7_reproducibility(ctx)]
    # G8 needs the card, which embeds G1-G7 outcomes.
    ctx.card_text = ctx.project.render_model_card({
        "candidate": ctx.candidate, "eval": ctx.eval, "champion_eval": ctx.champion_eval, "bundle": ctx.bundle,
        "gate_summary": [r.to_dict() for r in results], "provenance": prov, "report_dir": ctx.report_dir,
        "gate_config_hash": gate_config_hash,
    })
    ctx.report_dir.mkdir(parents=True, exist_ok=True)
    (ctx.report_dir / "MODEL_CARD.md").write_text(ctx.card_text, encoding="utf-8")
    results.append(g8_model_card(ctx))
    overall = "fail" if any(r.status == FAIL for r in results) else "pass"
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gate_report",
        "project": ctx.candidate.project,
        "family": ctx.candidate.family,
        "candidate_hash": ctx.candidate.hash,
        "champion_hash": ctx.champion.hash if ctx.champion else None,
        "created": utc_now(),
        "overall": overall,
        "failed_gates": [r.id for r in results if r.status == FAIL],
        "warnings": [f"{r.id}:{c.name}" for r in results for c in r.checks if c.status == "warn"],
        "provenance": prov,
        "gate_config_hash": gate_config_hash,
        "metrics": ctx.eval.metrics,
        "metric_ci": ctx.eval.metric_ci,
        "champion_metrics": ctx.champion_eval.metrics if ctx.champion_eval else None,
        "calibration": ctx.eval.calibration,
        "calibration_detail": ctx.eval.extra.get("calibration_detail"),
        "slices": ctx.eval.slices,
        "gates": [r.to_dict() for r in results],
        "synthetic": bool(ctx.bundle.meta.get("synthetic", False)),
    }
    (ctx.report_dir / "gate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    (ctx.report_dir / "gate_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def load_gate_report(report_dir: Path) -> dict[str, Any] | None:
    p = Path(report_dir) / "gate_report.json"
    return json.loads(p.read_text()) if p.exists() else None


def render_markdown(rep: dict[str, Any]) -> str:
    icon = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skip": "SKIP"}
    lines = [f"# Gate report for `{rep['candidate_hash']}`", "",
             f"**Overall: {icon[rep['overall']]}** | project `{rep['project']}` | family `{rep['family']}` | "
             f"data `{rep['provenance']['data_version']}` | gate config `{rep['gate_config_hash']}`", ""]
    for g in rep["gates"]:
        lines += [f"## {g['id']} {g['name']}: {icon[g['status']]}", "", "| check | status | value | threshold | CI | detail |", "|---|---|---|---|---|---|"]
        for c in g["checks"]:
            lines.append(f"| {c['name']} | {icon[c['status']]} | {c['value']} | {c['threshold']} | {c['ci']} | {c['detail']} |")
        lines.append("")
    return "\n".join(lines)
