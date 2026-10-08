"""Seeded bootstrap helpers (row-level resampling)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ScoreFn = Callable[[np.ndarray, np.ndarray, np.ndarray | None], dict[str, float]]


def _take(a: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
    return None if a is None else a[idx]


def bootstrap_scores(score: ScoreFn, y, pred, w, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Return {metric: array of n_boot bootstrap values}."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = score(y[idx], pred[idx], _take(w, idx))
        for k, v in s.items():
            out.setdefault(k, []).append(float(v))
    return {k: np.asarray(v) for k, v in out.items()}


def ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap(score: ScoreFn, y, pred_a, pred_b, w, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Bootstrap values of metric(a) - metric(b) on identical resamples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        wi = _take(w, idx)
        sa = score(y[idx], pred_a[idx], wi)
        sb = score(y[idx], pred_b[idx], wi)
        for k in sa:
            out.setdefault(k, []).append(float(sa[k] - sb[k]))
    return {k: np.asarray(v) for k, v in out.items()}
