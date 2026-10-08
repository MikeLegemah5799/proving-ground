"""Constant vs GLM vs GBM on the same holdout, with bootstrap CIs. Each candidate's holdout is read once (cached)."""

from __future__ import annotations

import json

from proving_ground.evaluation import evaluate_holdout


def compare(ws, hashes: dict[str, str]) -> dict:
    """hashes: {display name: candidate hash}. Writes reports/model_comparison.json."""
    b, rows = ws.bundle(), []
    n_boot = int(ws.gates_cfg().get("n_boot", 200))
    for name, h in hashes.items():
        cand = ws.registry.load(h)
        ev = evaluate_holdout(ws.project, b, cand, ws.report_dir(h), ws.reports_dir / "holdout_ledger.jsonl", n_boot, ws.seed)
        rows.append({"name": name, "hash": h, "poisson_deviance": ev.metrics["poisson_deviance"], "dev_ci": ev.metric_ci["poisson_deviance"],
                     "deviance_skill": ev.metrics["deviance_skill"], "gini": ev.metrics["gini"], "gini_ci": ev.metric_ci["gini"],
                     "decile_calibration_error": ev.calibration["decile_calibration_error"]})
    best = min(rows, key=lambda r: r["poisson_deviance"])
    note = (f"Lowest deviance: {best['name']}. Differences smaller than the CI width are not evidence of a real difference; "
            "see the paired-bootstrap G3 check for the champion/challenger decision. Holdout is one random grouped split, not a time split.")
    out = {"schema_version": "1.0", "kind": "model_comparison", "rows": rows, "note": note}
    ws.reports_dir.mkdir(parents=True, exist_ok=True)
    (ws.reports_dir / "model_comparison.json").write_text(json.dumps(out, indent=2))
    return out
