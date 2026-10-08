"""Constant baseline, Poisson GLM and LightGBM claim-frequency models (all predict a *rate*)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT = ["Area", "VehBrand", "VehGas", "Region"]
NUM = ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CAPS = {"VehPower": 12, "VehAge": 20, "DrivAge": 90, "BonusMalus": 150}
CAT_LEVELS = {
    "Area": list("ABCDEF"),
    "VehBrand": ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"],
    "VehGas": ["Diesel", "Regular"],
    "Region": ["R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43", "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"],
}
BINS = {  # GLM bins (left-closed); see docs/FEATURES.md for the reasoning
    "DrivAge": [18, 21, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    "VehAge": [0, 1, 2, 3, 5, 8, 11, 15, 20],
    "BonusMalus": [50, 51, 60, 70, 80, 95, 110, 130],
    "VehPower": [4, 5, 6, 7, 8, 9, 10, 11],
    # Claims are NOT proportional to exposure in this data (short terms show far higher frequency per exposure-year),
    # so exposure also enters as a covariate. See docs/FEATURES.md.
    "Exposure": [0.1, 0.25, 0.5, 0.75, 0.99],
}


class Binner(BaseEstimator, TransformerMixin):
    """Caps then bins selected numeric columns into string labels (unseen/NaN -> its own bin)."""

    def __init__(self, bins=None, caps=None):
        self.bins, self.caps = bins, caps

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        bins, caps = self.bins or BINS, self.caps or CAPS
        out = {}
        for c, edges in bins.items():
            v = pd.to_numeric(X[c], errors="coerce").clip(upper=caps.get(c))
            out[c] = np.digitize(v.fillna(-1).to_numpy(), edges).astype(str)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, names=None):
        return np.array(list(self.bins or BINS))


def _glm_pipeline(alpha: float, max_iter: int = 300) -> Pipeline:
    prep = ColumnTransformer([
        ("bins", Pipeline([("bin", Binner()), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), list(BINS)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT),
        ("dens", Pipeline([("log", _LogDensity()), ("sc", StandardScaler())]), ["Density"]),
    ], sparse_threshold=1.0)
    return Pipeline([("prep", prep), ("glm", PoissonRegressor(alpha=alpha, max_iter=max_iter, solver="lbfgs"))])


class _LogDensity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.clip(pd.to_numeric(X.iloc[:, 0], errors="coerce").fillna(1).to_numpy(), 1, 30000)).reshape(-1, 1)

    def get_feature_names_out(self, names=None):
        return np.array(["Density"])


class ConstantModel:
    def __init__(self, rate: float):
        self.rate = float(rate)

    def predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate)


class GLMModel:
    def __init__(self, alpha: float, scale: float = 1.0, corrupt: bool = False):
        self.alpha, self.scale, self.corrupt = alpha, scale, corrupt
        self.pipe: Pipeline | None = None

    def fit(self, X: pd.DataFrame, claims: np.ndarray, exposure: np.ndarray):
        self.pipe = _glm_pipeline(self.alpha)
        self.pipe.fit(X, claims / exposure, glm__sample_weight=exposure)
        return self

    def _x(self, X):
        if self.corrupt:  # simulated bug in preprocessing: two columns swapped at inference
            X = X.copy()
            X["DrivAge"], X["BonusMalus"] = X["BonusMalus"].to_numpy(), X["DrivAge"].to_numpy()
        return X

    def predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        return self.scale * self.pipe.predict(self._x(X))


class GBMModel:
    """LightGBM with a Poisson objective on the claim rate, weighted by exposure."""

    FEATS = [*NUM, *CAT]

    def __init__(self, params: dict, scale: float = 1.0, corrupt: bool = False):
        self.params, self.scale, self.corrupt = params, scale, corrupt
        self.booster = None

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.corrupt:
            X = X.copy()
            X["DrivAge"], X["BonusMalus"] = X["BonusMalus"].to_numpy(), X["DrivAge"].to_numpy()
        f = X[self.FEATS].copy()
        for c in CAT:
            f[c] = pd.Categorical(f[c].astype(str), categories=CAT_LEVELS[c])  # unseen level -> NaN (documented)
        return f

    def fit(self, X, claims, exposure, X_val=None, claims_val=None, exposure_val=None):
        import warnings

        import lightgbm as lgb

        p = dict(self.params)
        n_est = p.pop("n_estimators", 400)
        self.booster = lgb.LGBMRegressor(objective="poisson", n_estimators=n_est, verbose=-1, deterministic=True,
                                         force_row_wise=True, **p)
        kw = {}
        if X_val is not None:
            kw = dict(eval_set=[(self._frame(X_val), claims_val / exposure_val)], eval_sample_weight=[exposure_val],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.booster.fit(self._frame(X), claims / exposure, sample_weight=exposure, **kw)
        return self

    def predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        return self.scale * self.booster.predict(self._frame(X))
