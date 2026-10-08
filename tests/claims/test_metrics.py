from __future__ import annotations

import numpy as np
import pytest
from examples.claims_frequency import metrics as M


def test_poisson_deviance_hand_computed():
    # y=[1,0], mu=[2,1]: 2*[(1*ln(1/2) - (1-2)) + (0 - (0-1))] = 2*[(-0.693147+1) + 1] = 2.613706
    assert M.poisson_deviance_sum([1, 0], [2, 1]) == pytest.approx(2.6137056, rel=1e-6)
    # per exposure: rate=[2,1], exposure=[1,1] -> same / 2
    assert M.poisson_deviance([1, 0], [2, 1], [1, 1]) == pytest.approx(2.6137056 / 2, rel=1e-6)
    assert M.poisson_deviance_sum([3, 0, 2], [3, 0.0, 2]) == pytest.approx(0.0, abs=1e-9)


def test_gini_hand_computed():
    # perfect ordering of 4 equal-exposure policies, claims on the 2 highest-ranked: Lorenz y=[0,0,.5,1] -> area 0.25 -> gini 0.5
    assert M.gini([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], [1, 1, 1, 1]) == pytest.approx(0.5)
    assert M.gini([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4], [1, 1, 1, 1]) == pytest.approx(-0.5)
    assert M.gini([1, 0, 1, 0], [0.3, 0.3, 0.3, 0.3], [1, 1, 1, 1]) == pytest.approx(0.0)   # ties pooled


def test_exposure_matters_in_metrics():
    """Invariant 8: a metric that ignores exposure gives a detectably different answer on a constructed case."""
    y = np.array([1.0, 1.0, 0.0, 0.0])
    expo = np.array([0.1, 1.0, 1.0, 1.0])
    good = np.array([10.0, 1.0, 0.0001, 0.0001])        # rate high where exposure is tiny (1 claim in 0.1 yr)
    flat = np.full(4, y.sum() / expo.sum())
    assert M.poisson_deviance(y, good, expo) != pytest.approx(M.poisson_deviance(y, good, np.ones(4)), rel=1e-3)
    assert M.ae_ratio(y, flat, expo) == pytest.approx(1.0)
    assert M.ae_ratio(y, flat, np.ones(4)) != pytest.approx(1.0, rel=1e-2)


def test_exposure_is_respected_in_training(ws):
    """Training with exposure ignored (all weights 1, rate target = counts) must give different predictions."""
    from examples.claims_frequency.model import GLMModel
    b = ws.bundle()
    tr = b.train
    y, w = tr["ClaimNb"].to_numpy(float), tr["Exposure"].to_numpy(float)
    good = GLMModel(1e-4).fit(tr[b.features], y, w).predict_rate(b.holdout[b.features])
    ignorant = GLMModel(1e-4).fit(tr[b.features], y, np.ones_like(w)).predict_rate(b.holdout[b.features])
    assert abs(good.mean() - ignorant.mean()) / good.mean() > 0.1
    assert good.mean() == pytest.approx(y.sum() / w.sum(), rel=0.1)


def test_miscalibrated_predictor_has_larger_calibration_error():
    rng = np.random.default_rng(0)
    rate = rng.uniform(0.02, 0.3, 50000)
    expo = np.ones(50000)
    y = rng.poisson(rate * expo).astype(float)
    calibrated = M.decile_calibration_error(y, rate, expo)
    skewed = M.decile_calibration_error(y, rate * np.where(rate > 0.16, 1.6, 0.6), expo)
    assert skewed > 3 * calibrated and calibrated < 0.1
