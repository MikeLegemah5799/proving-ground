"""Cross-validated tuning INSIDE the training split only (the holdout is never read here)."""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import GroupKFold

from . import data as D
from . import metrics as M
from .model import GBMModel, GLMModel

GLM_ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2]
GBM_GRID = [
    {"num_leaves": 8, "min_child_samples": 800, "reg_lambda": 10.0, "colsample_bytree": 0.8},
    {"num_leaves": 16, "min_child_samples": 400, "reg_lambda": 5.0, "colsample_bytree": 0.8},
    {"num_leaves": 16, "min_child_samples": 1000, "reg_lambda": 20.0, "colsample_bytree": 0.7},
    {"num_leaves": 32, "min_child_samples": 600, "reg_lambda": 10.0, "colsample_bytree": 0.8},
    {"num_leaves": 24, "min_child_samples": 300, "reg_lambda": 2.0, "colsample_bytree": 0.9},
    {"num_leaves": 12, "min_child_samples": 1500, "reg_lambda": 30.0, "colsample_bytree": 0.7},
]


def tune(project, bundle, family: str, seed: int, n_folds: int = 3, max_rows: int = 150_000) -> dict:
    tr = bundle.train
    if len(tr) > max_rows:
        tr = tr.sample(n=max_rows, random_state=seed)
    tr = tr.reset_index(drop=True)
    groups = tr.groupby(D.PROFILE, sort=False).ngroup().to_numpy()
    folds = list(GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(tr, groups=groups))
    y, w = tr[bundle.target].to_numpy(float), tr[bundle.weight].to_numpy(float)
    t0 = time.perf_counter()
    log = {"family": family, "cv": f"{n_folds}-fold grouped by risk profile, inside the training split", "rows": len(tr), "trials": []}

    def cv(make):
        scores, extra = [], []
        for a, b in folds:
            m = make(tr.iloc[a], y[a], w[a], tr.iloc[b], y[b], w[b])
            scores.append(M.poisson_deviance(y[b], m.predict_rate(tr.iloc[b][bundle.features]), w[b]))
            extra.append(getattr(getattr(m, "booster", None), "best_iteration_", None))
        return float(np.mean(scores)), float(np.std(scores)), extra

    best, best_score = None, np.inf
    if family == "glm":
        for a in GLM_ALPHAS:
            mean, sd, _ = cv(lambda X, yy, ww, Xv, yv, wv, a=a: GLMModel(a).fit(X[bundle.features], yy, ww))
            log["trials"].append({"alpha": a, "cv_deviance_mean": mean, "cv_deviance_sd": sd})
            if mean < best_score:
                best, best_score = {"alpha": a}, mean
    else:
        for cfg in GBM_GRID:
            p = {"learning_rate": 0.08, "n_estimators": 800, "subsample": 0.8, "subsample_freq": 1, "num_threads": 4, "random_state": seed, **cfg}
            mean, sd, its = cv(lambda X, yy, ww, Xv, yv, wv, p=p: GBMModel(p).fit(
                X[bundle.features], yy, ww, Xv[bundle.features], yv, wv))
            its = [i for i in its if i]
            log["trials"].append({**cfg, "cv_deviance_mean": mean, "cv_deviance_sd": sd, "best_iterations": its})
            if mean < best_score:
                n_est = int(np.mean(its) * 1.1 * 0.08 / 0.05) if its else 600
                best, best_score = {**cfg, "n_estimators": max(100, min(1500, n_est))}, mean
    log.update(best=best, best_cv_deviance=best_score, seconds=round(time.perf_counter() - t0, 1), budget=f"{len(log['trials'])} configs x {n_folds} folds")
    return {"best": best, "log": log}
