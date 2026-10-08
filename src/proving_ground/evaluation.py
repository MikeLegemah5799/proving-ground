"""Holdout evaluation, performed exactly once per candidate and cached on disk.

Re-running gates with new thresholds reuses the cached predictions and metrics, so the
holdout is never re-scored for the same candidate (Invariant 10). A ledger records each
first-time evaluation; a test asserts one ledger line per candidate hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .candidate import Candidate
from .interface import DataBundle, ModelProject
from .stats import bootstrap_scores, ci


@dataclass
class HoldoutEval:
    candidate_hash: str
    data_version: str
    y: np.ndarray
    w: np.ndarray | None
    pred: np.ndarray
    metrics: dict[str, float]
    metric_ci: dict[str, list[float]]
    calibration: dict[str, float]
    slices: dict[str, list[dict[str, Any]]]
    extra: dict[str, Any] = field(default_factory=dict)


def _slice_rows(project: ModelProject, bundle: DataBundle, y, pred, w, n_boot: int, seed: int,
                reference_levels: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for sname, spec in project.slices().items():
        labels = spec.assign(bundle.holdout, reference_levels.get(sname)).to_numpy()
        rows = []
        for level in sorted(set(labels)):
            m = labels == level
            n = int(m.sum())
            row: dict[str, Any] = {"level": level, "n": n, "weight": float(w[m].sum()) if w is not None else float(n),
                                   "min_samples": spec.min_samples}
            if n < 2:
                row.update(scores={}, ci={})
            else:
                sc = project.score(y[m], pred[m], w[m] if w is not None else None)
                row["scores"] = {k: float(v) for k, v in sc.items()}
                if n >= spec.min_samples:
                    bs = bootstrap_scores(project.score, y[m], pred[m], w[m] if w is not None else None, n_boot, seed)
                    row["ci"] = {k: list(ci(v)) for k, v in bs.items()}
                else:
                    row["ci"] = {}
            rows.append(row)
        out[sname] = rows
    return out


def evaluate_holdout(project: ModelProject, bundle: DataBundle, cand: Candidate, report_dir: Path,
                     ledger: Path, n_boot: int = 200, seed: int = 7) -> HoldoutEval:
    cache_json, cache_npz = report_dir / "holdout_eval.json", report_dir / "holdout_eval.npz"
    if cache_json.exists() and cache_npz.exists():
        meta = json.loads(cache_json.read_text())
        if meta["data_version"] == bundle.data_version and meta["candidate_hash"] == cand.hash:
            z = np.load(cache_npz)
            return HoldoutEval(meta["candidate_hash"], meta["data_version"], z["y"], z["w"] if "w" in z else None,
                               z["pred"], meta["metrics"], meta["metric_ci"], meta["calibration"], meta["slices"],
                               meta.get("extra", {}))
    report_dir.mkdir(parents=True, exist_ok=True)
    ho = bundle.holdout
    pred = np.asarray(project.predict(cand.model, ho[bundle.features]), dtype=float)
    y = ho[bundle.target].to_numpy(dtype=float)
    w = ho[bundle.weight].to_numpy(dtype=float) if bundle.weight else None
    metrics = {k: float(v) for k, v in project.metrics(cand.model, bundle).items()}
    bs = bootstrap_scores(project.score, y, pred, w, n_boot, seed)
    metric_ci = {k: list(ci(v)) for k, v in bs.items()}
    calibration = {k: float(v) for k, v in project.calibration_report(cand.model, bundle).items()}
    ref_levels = {n: list(s.assign(bundle.train).value_counts().index[: s.top_k]) for n, s in project.slices().items() if s.top_k}
    slices = _slice_rows(project, bundle, y, pred, w, max(50, n_boot // 2), seed, ref_levels)
    extra: dict[str, Any] = {}
    if hasattr(project, "calibration_arrays"):
        extra["calibration_detail"] = project.calibration_arrays(y, pred, w)
    ev = HoldoutEval(cand.hash, bundle.data_version, y, w, pred, metrics, metric_ci, calibration, slices, extra)
    arrays = {"y": y, "pred": pred, **({"w": w} if w is not None else {})}
    np.savez_compressed(cache_npz, **arrays)
    cache_json.write_text(json.dumps({
        "candidate_hash": cand.hash, "data_version": bundle.data_version, "metrics": metrics,
        "metric_ci": metric_ci, "calibration": calibration, "slices": slices, "n_boot": n_boot, "extra": extra,
    }, indent=2, sort_keys=True))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as fh:
        fh.write(json.dumps({"candidate_hash": cand.hash, "data_version": bundle.data_version}) + "\n")
    return ev
