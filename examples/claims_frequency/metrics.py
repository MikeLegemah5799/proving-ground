"""Exposure-aware metrics. ``pred`` is a claim *rate* (claims per exposure-year); expected claims = pred * exposure.

All functions take y = claim counts and w = exposure, and are checked against hand-computed cases in tests.
"""

from __future__ import annotations

import numpy as np


def poisson_deviance_sum(y: np.ndarray, mu: np.ndarray) -> float:
    """2 * sum( y*ln(y/mu) - (y - mu) ), with 0*ln(0) = 0."""
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), 1e-12)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))


def poisson_deviance(y, rate, exposure) -> float:
    """Deviance per unit exposure (exposure-weighted mean deviance)."""
    w = np.asarray(exposure, float)
    return poisson_deviance_sum(y, np.asarray(rate, float) * w) / float(w.sum())


def gini(y, rate, exposure) -> float:
    """Exposure-weighted Gini: Lorenz curve of claims vs exposure, policies ordered by predicted rate (ties pooled).

    gini = 1 - 2 * area under the Lorenz curve; 0 for a non-discriminating model, higher is better."""
    y, rate, w = (np.asarray(a, float) for a in (y, rate, exposure))
    uniq, inv = np.unique(rate, return_inverse=True)
    yy, ww = np.bincount(inv, weights=y), np.bincount(inv, weights=w)
    cx = np.concatenate([[0.0], np.cumsum(ww) / ww.sum()])
    total = yy.sum()
    if total == 0:
        return 0.0
    cy = np.concatenate([[0.0], np.cumsum(yy) / total])
    area = float(np.sum((cx[1:] - cx[:-1]) * (cy[1:] + cy[:-1]) / 2))
    return 1.0 - 2.0 * area


def ae_ratio(y, rate, exposure) -> float:
    exp_claims = float(np.sum(np.asarray(rate, float) * np.asarray(exposure, float)))
    return float(np.sum(y)) / exp_claims if exp_claims > 0 else float("nan")


def decile_table(y, rate, exposure, n_bins: int = 10) -> list[dict]:
    """Equal-exposure risk deciles by predicted rate: predicted vs observed frequency."""
    y, rate, w = (np.asarray(a, float) for a in (y, rate, exposure))
    order = np.argsort(rate, kind="mergesort")
    y, rate, w = y[order], rate[order], w[order]
    cum = np.cumsum(w) / w.sum()
    bins = np.minimum((cum - 1e-12) * n_bins, n_bins - 1).astype(int)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        wb = float(w[m].sum())
        rows.append({"decile": b + 1, "exposure": wb, "policies": int(m.sum()), "claims": float(y[m].sum()),
                     "predicted": float((rate[m] * w[m]).sum() / wb), "observed": float(y[m].sum() / wb)})
    return rows


def decile_calibration_error(y, rate, exposure, n_bins: int = 10) -> float:
    """Exposure-weighted mean |observed - predicted| across deciles, relative to the portfolio frequency."""
    rows = decile_table(y, rate, exposure, n_bins)
    tot = sum(r["exposure"] for r in rows)
    base = sum(r["claims"] for r in rows) / tot
    return float(sum(r["exposure"] / tot * abs(r["observed"] - r["predicted"]) for r in rows) / base)
