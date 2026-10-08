"""Drift statistics (built-in; Evidently reports are generated in addition when available)."""

from __future__ import annotations

import numpy as np
from scipy import stats

EPS = 1e-4


def _psi(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.median(ref), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / max(len(cur), 1)
    return _psi(p, q)


def psi_categorical(ref, cur) -> float:
    cats = sorted(set(ref) | set(cur), key=str)
    rc = {c: 0 for c in cats}
    cc = {c: 0 for c in cats}
    for v in ref:
        rc[v] += 1
    for v in cur:
        cc[v] += 1
    return _psi(np.array([rc[c] for c in cats], float) / max(len(ref), 1), np.array([cc[c] for c in cats], float) / max(len(cur), 1))


def ks_stat(ref, cur) -> tuple[float, float]:
    r = stats.ks_2samp(ref, cur)
    return float(r.statistic), float(r.pvalue)


def chi2_p(ref, cur) -> float:
    cats = sorted(set(ref) | set(cur), key=str)
    table = np.array([[sum(1 for v in ref if v == c) for c in cats], [sum(1 for v in cur if v == c) for c in cats]])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(table)[1])
