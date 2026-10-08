"""Windowed drift monitoring over logs/requests.jsonl.

For each window it writes ``reports/drift/<run>/window_NN.json`` (schema-versioned, consumed by the
report viewer) and, when Evidently is installed, ``window_NN.html``. Levels: ok < watch < alert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..promote import read_jsonl
from ..stats import bootstrap_scores, ci
from ..workspace import Workspace
from . import stats as S

SCHEMA_VERSION = "1.0"
ORDER = {"ok": 0, "watch": 1, "alert": 2}


def level(value: float | None, thr: dict[str, float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "ok"
    return "alert" if value >= thr["alert"] else ("watch" if value >= thr["watch"] else "ok")


def worst(levels: list[str]) -> str:
    return max(levels, key=lambda lv: ORDER[lv], default="ok")


def window_frame(rows: list[dict[str, Any]], features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame over ALL logged rows (including ones serving rejected, so unseen categories and
    retyped columns stay visible to the monitor); predictions only exist for accepted rows."""
    df = pd.DataFrame([r["features"] for r in rows]) if rows else pd.DataFrame(columns=features)
    ok = [r for r in rows if not r.get("invalid")]
    return df, pd.Series([r["champion_pred"] for r in ok], dtype=float)


def feature_drift(ref: pd.Series, cur: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    numeric = pd.api.types.is_numeric_dtype(ref) and not pd.api.types.is_bool_dtype(ref)
    out: dict[str, Any] = {"type": "numeric" if numeric else "categorical", "n": int(len(cur))}
    if numeric:
        c = pd.to_numeric(cur, errors="coerce")
        bad_type = float(c.isna().mean()) if len(c) else 0.0
        c = c.dropna()
        out["psi"] = S.psi_numeric(ref.to_numpy(), c.to_numpy()) if len(c) else None
        ks, p = S.ks_stat(ref.to_numpy(), c.to_numpy()) if len(c) else (None, None)
        out.update(ks_stat=ks, ks_p=p, out_of_range_rate=float(((c < ref.min()) | (c > ref.max())).mean()) if len(c) else None,
                   new_category_rate=None, non_numeric_rate=bad_type, mean_ref=float(ref.mean()), mean_cur=float(c.mean()) if len(c) else None)
    else:
        known = set(ref.astype(str))
        cs = cur.astype(str)
        out["psi"] = S.psi_categorical(list(ref.astype(str)), list(cs))
        out.update(chi2_p=S.chi2_p(list(ref.astype(str)), list(cs)), new_category_rate=float((~cs.isin(known)).mean()),
                   out_of_range_rate=None, ks_stat=None, new_categories=sorted(set(cs) - known)[:10])
    fc = cfg["feature"]
    levels = {"psi": level(out["psi"], fc["psi"]), "new_category_rate": level(out.get("new_category_rate"), fc["new_category_rate"]),
              "out_of_range_rate": level(out.get("out_of_range_rate"), fc["out_of_range_rate"]),
              "ks_stat": level(out.get("ks_stat"), fc["ks_stat"]),
              "type_error_rate": level(out.get("non_numeric_rate"), cfg["validation_failure_rate"])}
    out["signal_levels"] = levels
    out["level"] = worst(list(levels.values()))
    return out


def monitor_windows(ws: Workspace, run_name: str = "latest", *, windows: list[int] | None = None,
                    html: bool = True) -> dict[str, Any]:
    cfg = ws.monitoring_cfg()
    b = ws.bundle()
    ref_df = b.reference[b.features]
    mon = ws.project.monitored_features()
    ref_cache: dict[str, tuple[pd.Series, dict[str, float]]] = {}

    def ref_for(h: str):
        """Reference predictions/scores come from the champion that actually served the window."""
        if h not in ref_cache:
            pr = pd.Series(np.asarray(ws.project.predict(ws.registry.load(h).model, ref_df), float))
            sc = ws.project.score(b.reference[b.target].to_numpy(float), pr.to_numpy(), b.reference[b.weight].to_numpy(float) if b.weight else None)
            ref_cache[h] = (pr, sc)
        return ref_cache[h]
    rows_all = [r for r in read_jsonl(ws.logs_dir / "requests.jsonl")]
    labels = {r["request_id"]: r["label"] for r in read_jsonl(ws.logs_dir / "labels.jsonl")}
    mname = cfg["label_drift"]["metric"]
    by_win: dict[int, list[dict[str, Any]]] = {}
    for r in rows_all:
        by_win.setdefault(int(r.get("window", 0)), []).append(r)
    out_dir = ws.reports_dir / "drift" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for win in sorted(by_win):
        if windows is not None and win not in windows:
            continue
        rows = by_win[win]
        champ_hash = max({r["champion_hash"] for r in rows}, key=lambda h: sum(1 for r in rows if r["champion_hash"] == h))
        ref_pred, ref_score = ref_for(champ_hash)
        cur_df, cur_pred = window_frame(rows, b.features)
        n_total, n_invalid = len(rows), sum(1 for r in rows if r.get("invalid"))
        syn = bool(rows and all(r.get("synthetic") for r in rows))
        scen = sorted({r.get("scenario") for r in rows if r.get("scenario")})
        feats: dict[str, Any] = {}
        if len(cur_df) >= min(cfg["window_min_rows"], 50):
            for f in mon:
                if f in cur_df.columns:
                    feats[f] = feature_drift(b.reference[f], cur_df[f], cfg)
        pred_psi = S.psi_numeric(ref_pred.to_numpy(), cur_pred.to_numpy()) if len(cur_pred) >= 30 else None
        pred = {"psi": pred_psi, "level": level(pred_psi, cfg["prediction"]["psi"]),
                "mean_ref": float(ref_pred.mean()), "mean_cur": float(cur_pred.mean()) if len(cur_pred) else None}
        vfr = n_invalid / n_total if n_total else 0.0
        val = {"failure_rate": vfr, "failures": n_invalid, "level": level(vfr, cfg["validation_failure_rate"])}
        lab = None
        lrows = [r for r in rows if not r.get("invalid") and r["request_id"] in labels]
        if len(lrows) >= 50:
            y = np.array([labels[r["request_id"]] for r in lrows])
            p = np.array([r["champion_pred"] for r in lrows])
            w = np.array([r["features"].get(b.weight, 1.0) for r in lrows], float) if b.weight else None
            sc = ws.project.score(y, p, w)
            rel = abs(sc[mname] - ref_score[mname]) / (abs(ref_score[mname]) or 1.0)
            bs = bootstrap_scores(ws.project.score, y, p, w, 200, 3)
            lo, hi = ci(bs[mname], 1 - cfg['label_drift'].get('confidence', 0.99))
            significant = not (lo <= ref_score[mname] <= hi)   # small windows are noisy: only a significant change counts
            lvl = level(rel, cfg["label_drift"]) if significant else "ok"
            lab = {"metric": mname, "value": float(sc[mname]), "reference": float(ref_score[mname]), "relative_change": float(rel),
                   "ci": [lo, hi], "significant": bool(significant), "n": len(lrows), "level": lvl,
                   "scores": {k: float(v) for k, v in sc.items()}}
        feat_alerts = sum(1 for v in feats.values() if v["level"] == "alert")
        levels = [v["level"] for v in feats.values()] + [pred["level"], val["level"]] + ([lab["level"]] if lab else [])
        overall = worst(levels)
        if overall == "alert" and feat_alerts < cfg.get("min_alert_features", 1) and pred["level"] != "alert" and val["level"] != "alert" and not (lab and lab["level"] == "alert"):
            overall = "watch"
        enough = n_total >= min(cfg["window_min_rows"], 50)
        summary = {
            "schema_version": SCHEMA_VERSION, "kind": "drift_summary", "run": run_name, "window": win, "level": overall if enough else "ok",
            "rows": n_total, "champion_hash": champ_hash, "synthetic": syn, "scenarios": scen, "features": feats,
            "prediction": pred, "validation": val, "label_drift": lab, "insufficient_data": not enough,
            "reference_rows": len(b.reference), "monitoring_config_hash": ws.monitoring_config_hash(),
            "window_start": min((r["ts"] for r in rows if r.get("ts")), default=None), "window_end": max((r["ts"] for r in rows if r.get("ts")), default=None), "html": None,
            "rollback_recommended": overall == "alert" and enough,
        }
        if html:
            hp = _evidently_html(ref_df, cur_df, mon, out_dir / f"window_{win:02d}.html")
            summary["html"] = hp.name if hp else None
        (out_dir / f"window_{win:02d}.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
        summaries.append(summary)
    timeline = {"schema_version": SCHEMA_VERSION, "kind": "drift_timeline", "run": run_name, "synthetic": all(s["synthetic"] for s in summaries) if summaries else False,
                "windows": [{"window": s["window"], "level": s["level"], "scenarios": s["scenarios"], "rows": s["rows"], "window_start": s["window_start"], "window_end": s["window_end"],
                             "features": {k: v["level"] for k, v in s["features"].items()}, "prediction": s["prediction"]["level"],
                             "validation": s["validation"]["level"], "label": s["label_drift"]["level"] if s["label_drift"] else None,
                             "file": f"window_{s['window']:02d}.json", "html": s["html"]} for s in summaries]}
    (out_dir / "timeline.json").write_text(json.dumps(timeline, indent=2, sort_keys=True))
    return timeline


def _evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str], path: Path) -> Path | None:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except Exception:
        return None
    try:
        cols = [c for c in cols if c in cur.columns]
        if len(cur) < 30 or not cols:
            return None
        snap = Report([DataDriftPreset(columns=cols)]).run(cur[cols].head(5000).reset_index(drop=True), ref[cols].head(5000).reset_index(drop=True))
        snap.save_html(str(path))
        return path
    except Exception:
        return None


def recommend(ws: Workspace, timeline: dict[str, Any], run_name: str = "latest", *, synthetic: bool = True) -> dict[str, Any] | None:
    """On the first `alert` window, write a `hold` audit entry carrying the rollback recommendation."""
    alerts = [w for w in timeline["windows"] if w["level"] == "alert"]
    if not alerts:
        return None
    first = alerts[0]
    champ = ws.registry.get_alias("champion")
    prev = ws.registry.get_alias("previous_champion")
    rec = f"Rollback to previous champion {prev} recommended." if prev else "No previous champion exists; investigate before continuing."
    drivers = [k for k, v in first["features"].items() if v == "alert"]
    if first["prediction"] == "alert":
        drivers.append("prediction")
    if first["validation"] == "alert":
        drivers.append("validation_failures")
    if first["label"] == "alert":
        drivers.append("label_drift")
    entry = ws.audit.append(
        "hold", candidate_hash=champ, champion_hash_before=champ, champion_hash_after=champ, git_sha=ws.prov()["git_sha"],
        data_version=ws.bundle().data_version, gate_config_hash=ws.gate_config_hash(), synthetic=synthetic,
        reason=f"Drift alert in window {first['window']} (drivers: {', '.join(drivers)}). {rec}",
        evidence=[{"kind": "drift_summary", "path": f"drift/{run_name}/{first['file']}", "summary": f"level alert; drivers {drivers}"}])
    if ws.monitoring_cfg().get("auto_rollback"):
        from ..rollback import rollback
        rollback(ws, reason=f"auto_rollback after drift alert in window {first['window']}", synthetic=synthetic, actor="monitor")
    return entry
