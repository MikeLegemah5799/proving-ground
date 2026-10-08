"""A trivial ModelProject used only by the template's own tests.

Target: y = 2*x1 - x2 + category effect + noise, weight column ``w``. Model: ridge regression.
Nothing in src/proving_ground imports this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.linear_model import Ridge

from proving_ground.interface import DataBundle, SliceSpec

CATS = ["a", "b", "c"]
FEATURES = ["x1", "x2", "cat", "w"]


def make_frame(n: int, seed: int, id_start: int = 0, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0 + shift, 1, n)
    x2 = rng.normal(0, 1, n)
    cat = rng.choice(CATS, n, p=[0.5, 0.4, 0.1])
    eff = pd.Series(cat).map({"a": 0.0, "b": 0.5, "c": -0.5}).to_numpy()
    y = 5 + 2 * x1 - x2 + eff + rng.normal(0, 0.5, n)
    return pd.DataFrame({"id": np.arange(id_start, id_start + n), "x1": x1, "x2": x2, "cat": cat,
                         "w": rng.uniform(0.5, 1.5, n), "y": y})


class ToyProject:
    name = "toy"

    def __init__(self, options: dict | None = None):
        self.options = options or {}
        self.degrade = bool(self.options.get("degrade", False))

    def load_data(self, data_version: str) -> DataBundle:
        n = int(self.options.get("n", 6000))
        df = make_frame(n * 2, seed=11)
        train, val, ho, stream = (df.iloc[:n], df.iloc[n:n + n // 4], df.iloc[n + n // 4:n + n // 2],
                                  df.iloc[n + n // 2:])
        return DataBundle(train=train.reset_index(drop=True), val=val.reset_index(drop=True),
                          holdout=ho.reset_index(drop=True), reference=val.reset_index(drop=True),
                          stream=stream.reset_index(drop=True), target="y", features=FEATURES,
                          data_version=data_version, weight="w", id_col="id", meta={"synthetic": True})

    def schema(self) -> pa.DataFrameSchema:
        return pa.DataFrameSchema({
            "x1": pa.Column(float, pa.Check.in_range(-10, 10)),
            "x2": pa.Column(float, pa.Check.in_range(-10, 10)),
            "cat": pa.Column(str, pa.Check.isin(CATS)),
            "w": pa.Column(float, pa.Check.in_range(0.01, 10)),
        }, strict=False, coerce=False)

    @staticmethod
    def _X(df: pd.DataFrame, zero_x1: bool = False) -> np.ndarray:
        x1 = np.zeros(len(df)) if zero_x1 else df["x1"]
        return np.column_stack([x1, df["x2"], (df["cat"] == "b"), (df["cat"] == "c")]).astype(float)

    def train(self, bundle: DataBundle, seed: int, params: dict) -> Any:
        df = bundle.train
        bad = self.degrade or params.get("degrade")  # simulates a corrupted preprocessing step
        m = Ridge(alpha=float(params.get("alpha", 1.0)))
        m.fit(self._X(df, bad), df["y"], sample_weight=df["w"])
        m.zero_x1 = bool(bad)
        return m

    def predict(self, model: Any, X: pd.DataFrame) -> pd.Series:
        return pd.Series(model.predict(self._X(X, getattr(model, 'zero_x1', False))), index=X.index)

    def score(self, y, pred, weight):
        w = np.ones_like(y) if weight is None else weight
        mse = float(np.sum(w * (y - pred) ** 2) / w.sum())
        var = float(np.sum(w * (y - np.average(y, weights=w)) ** 2) / w.sum())
        ratio = float(np.sum(w * y) / np.sum(w * pred)) if np.sum(w * pred) else float("nan")
        return {"mse": mse, "r2": 1 - mse / var if var else 0.0, "ae_ratio": ratio}

    def metric_directions(self):
        return {"mse": "lower", "r2": "higher", "ae_ratio": "lower"}

    def metrics(self, model, bundle):
        ho = bundle.holdout
        s = self.score(ho["y"].to_numpy(), self.predict(model, ho[FEATURES]).to_numpy(), ho["w"].to_numpy())
        return {"mse": s["mse"], "r2": s["r2"]}

    def slices(self):
        return {"cat": SliceSpec("cat", "categorical", min_samples=200)}

    def calibration_report(self, model, bundle):
        ho = bundle.holdout
        pred = self.predict(model, ho[FEATURES]).to_numpy()
        return {"calibration_error": float(abs(np.average(ho["y"], weights=ho["w"]) / np.average(pred, weights=ho["w"]) - 1))}

    def monitored_features(self):
        return ["x1", "x2", "cat"]

    def default_params(self, family: str) -> dict:
        return {"alpha": 1.0}

    def render_model_card(self, ctx: dict) -> str:
        c = ctx["candidate"]
        return (f"# Model card: toy {c.hash}\n\n## Intended use\nTests only.\n\n## Out-of-scope use\nEverything else.\n\n"
                f"## Data\nSynthetic.\n\n## Training procedure\nRidge.\n\n## Metrics\n{ctx['eval'].metrics}\n\n"
                "## Slices and calibration\nSee gate report.\n\n## Explainability\nLinear coefficients.\n\n"
                "## Limitations\nToy.\n\n## Provenance\n" + str(ctx["provenance"]) + "\n")
