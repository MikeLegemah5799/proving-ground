"""The adoption contract: implement ``ModelProject`` to plug a model into the template."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pandera.pandas as pa


@dataclass
class DataBundle:
    """Splits plus a reference sample. Splits must not share ``id_col`` values."""

    train: pd.DataFrame
    val: pd.DataFrame
    holdout: pd.DataFrame
    reference: pd.DataFrame            # unseen-by-training sample used as the monitoring baseline
    target: str
    features: list[str]                # raw input columns the serving API accepts
    data_version: str
    weight: str | None = None          # exposure / sample weight column, if any
    id_col: str | None = None
    stream: pd.DataFrame | None = None  # held-out rows replayed as *simulated* production traffic
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SliceSpec:
    column: str
    kind: Literal["categorical", "bins"] = "categorical"
    bins: list[float] | None = None     # edges for kind="bins" (open ended: -inf/inf added)
    top_k: int | None = None            # categorical: keep the k most frequent levels, pool the rest
    min_samples: int = 500

    def assign(self, df: pd.DataFrame, reference_levels: list[str] | None = None) -> pd.Series:
        col = df[self.column]
        if self.kind == "bins":
            assert self.bins, "bins required for kind='bins'"
            edges = [-np.inf, *self.bins, np.inf]
            labels = []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                if np.isinf(lo):
                    labels.append(f"<{hi:g}")
                elif np.isinf(hi):
                    labels.append(f">={lo:g}")
                else:
                    labels.append(f"{lo:g}-{hi:g}")
            return pd.cut(col, bins=edges, labels=labels, right=False).astype(str)
        s = col.astype(str)
        if self.top_k is not None:
            keep = reference_levels or list(s.value_counts().index[: self.top_k])
            s = s.where(s.isin(keep), "other")
        return s


@runtime_checkable
class ModelProject(Protocol):
    """Touchpoint 1. The first eight members are the spec'd contract; the last four are
    small extensions the gate suite needs (documented in TEMPLATE_GUIDE.md)."""

    name: str

    def load_data(self, data_version: str) -> DataBundle: ...
    def schema(self) -> pa.DataFrameSchema: ...
    def train(self, bundle: DataBundle, seed: int, params: dict) -> Any: ...
    def predict(self, model: Any, X: pd.DataFrame) -> pd.Series: ...
    def metrics(self, model: Any, bundle: DataBundle) -> dict[str, float]: ...
    def slices(self) -> dict[str, SliceSpec]: ...
    def calibration_report(self, model: Any, bundle: DataBundle) -> dict[str, float]: ...
    def monitored_features(self) -> list[str]: ...

    # --- extensions -------------------------------------------------------------
    def score(self, y: np.ndarray, pred: np.ndarray, weight: np.ndarray | None) -> dict[str, float]:
        """Array-level metrics; used for bootstrap CIs, slices and label-drift monitoring."""
        ...

    def metric_directions(self) -> dict[str, Literal["higher", "lower"]]: ...
    def render_model_card(self, ctx: dict[str, Any]) -> str: ...
    def default_params(self, family: str) -> dict: ...
