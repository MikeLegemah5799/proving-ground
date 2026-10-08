"""ClaimsFrequencyProject: the worked example's implementation of the ModelProject interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from proving_ground.interface import DataBundle, SliceSpec

from . import data as D
from . import metrics as M
from .card import render_card
from .model import CAT_LEVELS, ConstantModel, GBMModel, GLMModel
from .scenarios import claims_scenarios

TARGET, WEIGHT, ID = "ClaimNb", "Exposure", "IDpol"


class ClaimsFrequencyProject:
    name = "claims_frequency"

    def __init__(self, options: dict | None = None):
        self.options = options or {}
        self.base_rate: float | None = None
        self._data_version = "unloaded"

    # ------------------------------------------------------------------ data
    def data_version(self) -> str:
        return self._data_version if self._data_version != "unloaded" else self._resolve()[1]

    def _resolve(self) -> tuple[pd.DataFrame, str, bool]:
        src = self.options.get("data_source", "auto")
        if src == "auto":
            src = "openml" if D.CLEAN_FILE.exists() else "synthetic"
        if src == "openml":
            if not D.CLEAN_FILE.exists():
                raise SystemExit("Clean data not found. Run `make data` (downloads once, then works offline).")
            df, prefix, syn = pd.read_parquet(D.CLEAN_FILE), "fremtpl2", False
        else:
            if not D.SYN_FILE.exists():
                D.make_synthetic_file(int(self.options.get("synthetic_rows", 60000)))
            df, prefix, syn = pd.read_parquet(D.SYN_FILE), "synthetic", True
        sample = self.options.get("sample")
        if sample and sample < len(df):
            df = df.sample(n=int(sample), random_state=0).reset_index(drop=True)
        return df, D.data_version_of(df, prefix) + (f"-n{sample}" if sample else ""), syn

    def load_data(self, data_version: str | None = None) -> DataBundle:
        df, version, syn = self._resolve()
        self._data_version = version
        parts = D.split_groups(df, seed=int(self.options.get("split_seed", 42)))
        self.base_rate = float(parts["train"][TARGET].sum() / parts["train"][WEIGHT].sum())
        ref = parts["val"].sample(n=min(20000, len(parts["val"])), random_state=1).reset_index(drop=True)
        return DataBundle(
            train=parts["train"], val=parts["val"], holdout=parts["holdout"], reference=ref, stream=parts["stream"],
            target=TARGET, weight=WEIGHT, id_col=ID, features=D.FEATURES, data_version=version,
            meta={"synthetic": syn, "rows": {k: len(v) for k, v in parts.items()}, "base_rate": self.base_rate,
                  "split": "policy- and risk-profile-grouped, claim-stratified; NO temporal split (dataset has no dates)"})

    def schema(self) -> pa.DataFrameSchema:
        cat = lambda c: pa.Column(str, pa.Check.isin(CAT_LEVELS[c]))  # noqa: E731
        return pa.DataFrameSchema({
            "Exposure": pa.Column(float, pa.Check.in_range(0.0, 1.0, include_min=False)),
            "Area": cat("Area"), "VehBrand": cat("VehBrand"), "VehGas": cat("VehGas"), "Region": cat("Region"),
            "VehPower": pa.Column(int, pa.Check.in_range(1, 30)),
            "VehAge": pa.Column(int, pa.Check.in_range(0, 100)),
            "DrivAge": pa.Column(int, pa.Check.in_range(18, 100)),
            "BonusMalus": pa.Column(int, pa.Check.in_range(50, 350)),
            "Density": pa.Column(int, pa.Check.in_range(1, 100000)),
        }, strict=False, coerce=False)

    # ------------------------------------------------------------------ model
    def default_params(self, family: str) -> dict:
        return {
            "constant": {},
            "glm": {"alpha": 1e-4, "recalibrate": True},
            "gbm": {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 16, "min_child_samples": 400, "subsample": 0.8,
                    "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 5.0, "num_threads": 4, "recalibrate": True},
            "gbm_degraded": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 16, "min_child_samples": 400, "num_threads": 4,
                             "recalibrate": False, "corrupt_preprocessing": True},
        }[family]

    def train(self, bundle: DataBundle, seed: int, params: dict) -> Any:
        fam = params["_family"]
        tr = bundle.train
        y, w = tr[TARGET].to_numpy(float), tr[WEIGHT].to_numpy(float)
        if fam == "constant":
            return ConstantModel(y.sum() / w.sum())
        p = {k: v for k, v in params.items() if k not in ("_family", "recalibrate", "corrupt_preprocessing", "alpha")}
        corrupt = bool(params.get("corrupt_preprocessing"))
        if fam == "glm":
            model = GLMModel(float(params["alpha"])).fit(tr[bundle.features], y, w)
        else:
            va = bundle.val
            model = GBMModel({**p, "random_state": seed}).fit(tr[bundle.features], y, w, va[bundle.features],
                                                              va[TARGET].to_numpy(float), va[WEIGHT].to_numpy(float))
        if params.get("recalibrate"):  # scale factor fitted on the validation split, part of the candidate
            va = bundle.val
            exp_claims = float((model.predict_rate(va[bundle.features]) * va[WEIGHT]).sum())
            model.scale = float(va[TARGET].sum() / exp_claims)
        model.corrupt = corrupt
        return model

    def predict(self, model: Any, X: pd.DataFrame) -> pd.Series:
        return pd.Series(model.predict_rate(X), index=X.index)

    # ------------------------------------------------------------------ metrics
    def score(self, y, pred, weight) -> dict[str, float]:
        w = np.ones_like(y, dtype=float) if weight is None else weight
        dev = M.poisson_deviance(y, pred, w)
        base = self.base_rate if self.base_rate is not None else float(y.sum() / w.sum())
        dev0 = M.poisson_deviance(y, np.full_like(w, base, dtype=float), w)
        return {"poisson_deviance": dev, "deviance_skill": 1.0 - dev / dev0, "gini": M.gini(y, pred, w), "ae_ratio": M.ae_ratio(y, pred, w)}

    def metric_directions(self):
        return {"poisson_deviance": "lower", "deviance_skill": "higher", "gini": "higher"}

    def metrics(self, model: Any, bundle: DataBundle) -> dict[str, float]:
        ho = bundle.holdout
        s = self.score(ho[TARGET].to_numpy(float), self.predict(model, ho[bundle.features]).to_numpy(), ho[WEIGHT].to_numpy(float))
        return {k: s[k] for k in ("poisson_deviance", "deviance_skill", "gini")}

    def calibration_report(self, model: Any, bundle: DataBundle) -> dict[str, float]:
        ho = bundle.holdout
        y, w = ho[TARGET].to_numpy(float), ho[WEIGHT].to_numpy(float)
        p = self.predict(model, ho[bundle.features]).to_numpy()
        return {"decile_calibration_error": M.decile_calibration_error(y, p, w), "ae_ratio": M.ae_ratio(y, p, w)}

    def calibration_arrays(self, y, pred, w) -> dict:
        w = np.ones_like(y, dtype=float) if w is None else w
        return {"ae_ratio": M.ae_ratio(y, pred, w), "decile_calibration_error": M.decile_calibration_error(y, pred, w),
                "deciles": M.decile_table(y, pred, w)}

    def decile_table(self, model: Any, df: pd.DataFrame) -> list[dict]:
        p = self.predict(model, df[D.FEATURES]).to_numpy()
        return M.decile_table(df[TARGET].to_numpy(float), p, df[WEIGHT].to_numpy(float))

    def slices(self) -> dict[str, SliceSpec]:
        n = 2000
        return {
            "driver_age": SliceSpec("DrivAge", "bins", [25, 35, 45, 55, 65, 75], min_samples=n),
            "vehicle_age": SliceSpec("VehAge", "bins", [1, 3, 6, 10, 15], min_samples=n),
            "region": SliceSpec("Region", "categorical", top_k=8, min_samples=n),
            "bonus_malus": SliceSpec("BonusMalus", "bins", [55, 70, 90, 110], min_samples=n),
            "exposure": SliceSpec("Exposure", "bins", [0.25, 0.75, 1.0], min_samples=n),
        }

    def monitored_features(self) -> list[str]:
        return ["DrivAge", "VehAge", "VehPower", "BonusMalus", "Density", "Area", "VehBrand", "VehGas", "Region"]

    def drift_scenarios(self):
        return claims_scenarios()

    # ------------------------------------------------------------------ tuning
    def tune(self, bundle: DataBundle, family: str, seed: int, out_dir: Path | None = None) -> dict:
        from .tuning import tune
        res = tune(self, bundle, family, seed)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"tuning_{family}.json").write_text(json.dumps(res["log"], indent=2))
        return res["best"]

    def render_model_card(self, ctx: dict) -> str:
        return render_card(self, ctx)
