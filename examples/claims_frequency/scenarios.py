"""Insurance-specific drift scenarios (all synthetic). Expected outcomes are verified in tests."""

from __future__ import annotations

import numpy as np

from proving_ground.monitoring.injection import Scenario

from .model import CAT_LEVELS


def claims_scenarios() -> dict[str, Scenario]:
    def young_w(df, k):
        # over-sample real young drivers: features and labels stay consistent (real records, different mix)
        return 1.0 + 30.0 * k * (df["DrivAge"].to_numpy() < 25) + 8.0 * k * ((df["DrivAge"].to_numpy() >= 25) & (df["DrivAge"].to_numpy() < 30))

    def decay_w(df, k):
        return np.exp(0.12 * k * np.minimum(df["VehAge"].to_numpy(), 25))

    def brand(df, k, rng):
        m = rng.random(len(df)) < 0.15 * k
        df["VehBrand"] = df["VehBrand"].astype(object)
        df.loc[m, "VehBrand"] = "B99"          # a brand the model has never seen
        return df

    def bm_break(df, k, rng):
        df["BonusMalus"] = df["BonusMalus"].astype(str) + "%"   # upstream retyped the column
        return df

    def freq_shift(y, df, k, rng):
        return y + rng.poisson(0.5 * k * df["Exposure"].to_numpy() * max(float(y.sum() / df["Exposure"].sum()), 1e-9))

    assert "B99" not in CAT_LEVELS["VehBrand"]
    return {
        "young_driver_surge": Scenario("young_driver_surge", "A younger cohort enters the book: young drivers are over-represented in traffic", None,
                                       resample_fn=young_w, ramp=[0.0, 0.04, 0.09, 0.2, 0.5, 1.0], expected={"level": "alert", "signal": "feature:DrivAge"}),
        "new_vehicle_brand": Scenario("new_vehicle_brand", "Unseen VehBrand values appear", brand,
                                      expected={"level": "alert", "signal": "feature:VehBrand",
                                                "note": "serving rejects unknown brands via the schema (validation failures rise too); "
                                                        "the monitor still sees the raw values and reports the new-category rate"}),
        "bonus_malus_schema_break": Scenario("bonus_malus_schema_break", "BonusMalus arrives as a string", bm_break,
                                             expected={"level": "alert", "signal": "validation"}),
        "claims_frequency_shift": Scenario("claims_frequency_shift", "Delayed labels show a higher base claim rate", None, freq_shift,
                                           expected={"level": "alert", "signal": "label"}),
        "silent_decay": Scenario("silent_decay", "Vehicle age drifts up slowly (older vehicles gradually over-represented)", None,
                                 resample_fn=decay_w, ramp=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0], expected={"level": "alert", "signal": "feature:VehAge", "watch_before_alert": True}),
    }
