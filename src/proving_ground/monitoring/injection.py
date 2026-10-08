"""Synthetic drift injection. Every scenario is tagged synthetic and declares its expected outcome.

A scenario transforms the *features* (and optionally the delayed *labels*) of simulated traffic.
Expected outcomes are verified by tests; misses are documented, not hidden.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FeatFn = Callable[[pd.DataFrame, float, np.random.Generator], pd.DataFrame]
LabelFn = Callable[[np.ndarray, pd.DataFrame, float, np.random.Generator], np.ndarray]


@dataclass
class Scenario:
    name: str
    description: str
    feature_fn: FeatFn | None = None
    label_fn: LabelFn | None = None
    # Optional: sampling weights over the stream pool at a given intensity. Used for covariate shift that
    # keeps features and (real) labels consistent, by over-sampling a real sub-population.
    resample_fn: Callable[[pd.DataFrame, float], np.ndarray] | None = None
    # expected: signal kind that must reach the level at full intensity, e.g.
    # {"level": "alert", "signal": "feature:DrivAge"|"new_category"|"validation"|"label"|"prediction", "watch_before_alert": bool}
    expected: dict[str, Any] = field(default_factory=dict)
    ramp: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 1.0])
    synthetic: bool = True

    def resample_weights(self, df, intensity):
        return self.resample_fn(df, intensity) if self.resample_fn else None

    def apply_features(self, df, intensity, rng):
        return self.feature_fn(df.copy(), intensity, rng) if self.feature_fn else df

    def apply_labels(self, y, df, intensity, rng):
        return self.label_fn(y, df, intensity, rng) if self.label_fn else y


def generic_scenarios(numeric: str, categorical: str, weight: str | None = None) -> dict[str, Scenario]:
    """Domain-agnostic scenarios built from one numeric and one categorical monitored feature."""

    def covariate(df, k, rng):
        sd = float(df[numeric].std() or 1.0)
        df[numeric] = df[numeric] + 2.0 * sd * k
        return df

    def new_cat(df, k, rng):
        m = rng.random(len(df)) < 0.25 * k
        df[categorical] = df[categorical].astype(object)
        df.loc[m, categorical] = "UNSEEN_LEVEL"
        return df

    def schema_break(df, k, rng):
        df[numeric] = df[numeric].astype(str) + "x"
        return df

    def label_shift(y, df, k, rng):
        if np.allclose(y, np.round(y)):  # count-like labels: add extra events
            return y + rng.poisson(0.6 * k * max(float(np.mean(y)), 1e-9), size=len(y))
        return y * (1 + 0.6 * k)

    def silent(df, k, rng):
        sd = float(df[numeric].std() or 1.0)
        df[numeric] = df[numeric] + 0.6 * sd * k
        return df

    return {
        "covariate_shift": Scenario("covariate_shift", f"{numeric} shifts upward by ~2 standard deviations", covariate,
                                    expected={"level": "alert", "signal": f"feature:{numeric}"}),
        "new_category": Scenario("new_category", f"unseen levels appear in {categorical}", new_cat,
                                 expected={"level": "alert", "signal": f"feature:{categorical}"}),
        "schema_break": Scenario("schema_break", f"{numeric} is sent as a string", schema_break,
                                 expected={"level": "alert", "signal": "validation"}),
        "label_shift": Scenario("label_shift", "delayed labels show a higher base rate", None, label_shift,
                                expected={"level": "alert", "signal": "label"}),
        "silent_decay": Scenario("silent_decay", f"{numeric} drifts slowly", silent,
                                 expected={"level": "alert", "signal": f"feature:{numeric}", "watch_before_alert": True}),
    }
