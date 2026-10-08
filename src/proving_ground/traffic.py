"""Simulated production traffic. Everything sent here is tagged ``synthetic: true``.

Held-out policies/rows (``bundle.stream``) are replayed in random order. Labels arrive later
(written to logs/labels.jsonl) to model delayed feedback.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .workspace import Workspace


def _records(df: pd.DataFrame, features: list[str], id_col: str | None) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(df.to_dict("records")):
        rec = {c: _py(row[c]) for c in features if c in row}
        rec["key"] = str(row[id_col]) if id_col else str(i)
        out.append(rec)
    return out


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def run_traffic(ws: Workspace, client, *, n_windows: int, per_window: int, seed: int = 0, scenario=None,
                intensities: list[float] | None = None, batch_size: int = 100, start_window: int = 0,
                labels: bool = True, scenario_name: str | None = None) -> list[dict[str, Any]]:
    """Send ``n_windows`` windows of ``per_window`` requests. Returns per-window send summaries."""
    b = ws.bundle()
    assert b.stream is not None, "project must provide bundle.stream for simulated traffic"
    rng = np.random.default_rng(seed + start_window)
    ws.logs_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for w in range(n_windows):
        win = start_window + w
        inten = intensities[w] if intensities else 0.0
        sw = scenario.resample_weights(b.stream, inten) if (scenario is not None and inten > 0) else None
        pool = b.stream.sample(n=per_window, random_state=int(rng.integers(1 << 31)), replace=True, weights=sw)
        pool = pool.reset_index(drop=True)
        send_df = pool.copy()
        y = pool[b.target].to_numpy(dtype=float)
        if scenario is not None and inten > 0:
            send_df = scenario.apply_features(send_df, inten, rng)
            y = scenario.apply_labels(y, pool, inten, rng)
        sent = invalid = 0
        label_rows = []
        for s in range(0, len(send_df), batch_size):
            chunk = send_df.iloc[s:s + batch_size]
            body = {"records": _records(chunk, b.features, b.id_col), "window": win, "synthetic": True,
                    "scenario": scenario_name if inten > 0 else None}
            r = client.post("/predict", json=body)
            r.raise_for_status()
            for j, p in enumerate(r.json()["predictions"]):
                sent += 1
                if "error" in p:
                    invalid += 1
                elif labels:
                    label_rows.append({"request_id": p["request_id"], "label": float(y[s + j]), "window": win, "synthetic": True})
        if label_rows:
            with (ws.logs_dir / "labels.jsonl").open("a") as fh:
                for r_ in label_rows:
                    fh.write(json.dumps(r_) + "\n")
        summaries.append({"window": win, "sent": sent, "invalid": invalid, "intensity": inten, "scenario": scenario_name})
    return summaries
