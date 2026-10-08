"""Explainability artifacts: permutation importance, TreeSHAP (GBM), GLM coefficients, partial dependence.

Computed on a *validation* sample (never the holdout). SHAP and importances describe the fitted model,
not causal effects in the world."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .model import NUM, GBMModel, GLMModel

PDP_FEATURES = ["DrivAge", "BonusMalus", "VehAge", "Density"]


def _dev(project, model, X, y, w):
    return M.poisson_deviance(y, model.predict_rate(X), w)


def permutation_importance(project, model, bundle, n=20000, seed=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    v = bundle.val.sample(n=min(n, len(bundle.val)), random_state=seed).reset_index(drop=True)
    X, y, w = v[bundle.features], v[bundle.target].to_numpy(float), v[bundle.weight].to_numpy(float)
    base = _dev(project, model, X, y, w)
    out = []
    for f in list(bundle.features):
        Xp = X.copy()
        Xp[f] = Xp[f].to_numpy()[rng.permutation(len(Xp))]
        out.append({"feature": f, "deviance_increase": _dev(project, model, Xp, y, w) - base})
    return sorted(out, key=lambda r: -r["deviance_increase"])


def partial_dependence(project, model, bundle, feature, n=3000, seed=0, points=8) -> list[dict]:
    v = bundle.val.sample(n=min(n, len(bundle.val)), random_state=seed).reset_index(drop=True)
    X = v[bundle.features].copy()
    grid = np.unique(np.quantile(X[feature], np.linspace(0.02, 0.98, points)).round(2)) if feature in NUM else sorted(X[feature].unique())[:points]
    rows = []
    for g in grid:
        Xg = X.copy()
        Xg[feature] = g
        rows.append({"value": float(g) if feature in NUM else str(g), "avg_predicted_rate": float(model.predict_rate(Xg).mean())})
    return rows


def shap_summary(model: GBMModel, bundle, n=2000, seed=0) -> dict:
    v = bundle.val.sample(n=min(n, len(bundle.val)), random_state=seed).reset_index(drop=True)
    contrib = model.booster.booster_.predict(model._frame(v[bundle.features]), pred_contrib=True)
    names = model.FEATS
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    local = []
    for i in range(3):
        c = contrib[i]
        top = np.argsort(-np.abs(c[:-1]))[:4]
        local.append({"row": {f: (v.loc[i, f].item() if hasattr(v.loc[i, f], "item") else v.loc[i, f]) for f in names},
                      "baseline_log_rate": float(c[-1]), "top_contributions_log_rate": {names[j]: round(float(c[j]), 4) for j in top}})
    return {"method": "TreeSHAP via LightGBM pred_contrib (log-rate scale)",
            "mean_abs_shap": sorted(({"feature": n_, "mean_abs": float(m)} for n_, m in zip(names, mean_abs, strict=True)), key=lambda r: -r["mean_abs"]),
            "local_examples": local}


def glm_coefficients(model: GLMModel, top=15) -> list[dict]:
    names = model.pipe.named_steps["prep"].get_feature_names_out()
    coef = model.pipe.named_steps["glm"].coef_
    order = np.argsort(-np.abs(coef))[:top]
    return [{"term": str(names[i]), "coef_log_rate": float(coef[i]), "rate_ratio": float(np.exp(coef[i]))} for i in order]


def get_or_compute(project, cand, bundle, report_dir: Path) -> dict:
    p = report_dir / "explain.json"
    if p.exists():
        return json.loads(p.read_text())
    m = cand.model
    out: dict = {"note": "Describes the fitted model, not causal effects. Computed on a validation sample, not the holdout."}
    if cand.family == "constant":
        out["importance"], out["pdp"] = [], {}
    else:
        out["importance"] = permutation_importance(project, m, bundle)
        out["pdp"] = {f: partial_dependence(project, m, bundle, f) for f in PDP_FEATURES}
        if isinstance(m, GBMModel):
            out["shap"] = shap_summary(m, bundle)
        if isinstance(m, GLMModel):
            out["glm_coefficients"] = glm_coefficients(m)
    report_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return out
