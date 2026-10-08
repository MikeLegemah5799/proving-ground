# Model card: claim frequency `gbm` / candidate `d5ca3193399bdeea`

## Intended use

Educational and portfolio demonstration of a *gated model release process* for a motor third-party-liability claim-frequency model (expected claims per exposure-year). It exists to exercise gates, shadow/canary, drift monitoring and rollback.

## Out-of-scope use

Not for pricing, underwriting, reserving, or any decision about real people. No claim of regulatory compliance, actuarial sign-off or fairness certification. Severity (claim cost) is not modeled, so nothing here estimates expected loss.

## Data

- Source: freMTPL2freq (French motor TPL), fetched from OpenML data id 41214 at setup, never committed. OpenML lists it as CC0; the original CASdatasets distribution is GPL, so treat redistribution rights as unsettled (see docs/DATA_CARD.md).
- Data version: `fremtpl2-162afa27d8`
- Rows per split: {'train': 406716, 'val': 67656, 'holdout': 102035, 'stream': 101606}
- Split: policy- and risk-profile-grouped, claim-stratified; NO temporal split (dataset has no dates)
- **Known gaps: the dataset has no timestamps.** A time-based split is impossible, so *temporal generalization was not evaluated*. Simulated production traffic replays held-out policies in random order and is **not** real production behavior.
- One row per policy; severity data not used; exposure is capped at 1 year, claim counts at 4 (row counts per cleaning rule are in the data-quality report).

## Training procedure

- Family: `gbm`; seed `20260101`; params: `{"n_estimators": 287, "learning_rate": 0.05, "num_leaves": 16, "min_child_samples": 400, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 5.0, "num_threads": 4, "recalibrate": true}`
- Target: claim rate (claims per exposure-year), Poisson loss, exposure as sample weight (equivalent to an offset). Tuning used grouped cross-validation inside the training split only; the holdout is evaluated once per candidate.
- Optional scale-factor recalibration is fitted on the validation split and is part of the candidate (and its hash).

## Metrics

Evaluated on the holdout, exposure-weighted, 95% bootstrap CIs (policies resampled).

| metric | value | 95% CI |
|---|---|---|
| deviance_skill | 0.11375 | [0.10522, 0.12211] |
| gini | 0.37811 | [0.36193, 0.39346] |
| poisson_deviance | 0.55276 | [0.54119, 0.56314] |

### Three-model comparison (same holdout)

| model | candidate | deviance | CI | skill vs constant | Gini | CI |
|---|---|---|---|---|---|---|
| Constant baseline | `ddc7f3e844d4eef7` | 0.62370 | [0.61002, 0.63563] | 0.00000 | 0.00000 | [0.00000, 0.00000] |
| Poisson GLM | `13f8c79bff1da465` | 0.57744 | [0.56609, 0.58851] | 0.07418 | 0.32755 | [0.31142, 0.34304] |
| LightGBM | `d5ca3193399bdeea` | 0.55276 | [0.54119, 0.56314] | 0.11375 | 0.37811 | [0.36193, 0.39346] |

Lowest deviance: LightGBM. Differences smaller than the CI width are not evidence of a real difference; see the paired-bootstrap G3 check for the champion/challenger decision. Holdout is one random grouped split, not a time split.

## Slices and calibration

Calibration (exposure-weighted decile error, relative to portfolio frequency): **0.04324**; overall actual/expected: **1.00636**.

| slice | level | n | A/E | 95% CI | flag |
|---|---|---|---|---|---|
| bonus_malus | 55-70 | 16383 | 1.024 | [0.961, 1.073] |  |
| bonus_malus | 70-90 | 12690 | 1.029 | [0.957, 1.087] |  |
| bonus_malus | 90-110 | 8404 | 0.963 | [0.893, 1.040] |  |
| bonus_malus | <55 | 63670 | 1.000 | [0.956, 1.036] |  |
| bonus_malus | >=110 | 888 | 1.090 |  | insufficient data |
| driver_age | 25-35 | 21246 | 1.004 | [0.949, 1.063] |  |
| driver_age | 35-45 | 25669 | 1.003 | [0.946, 1.047] |  |
| driver_age | 45-55 | 24482 | 1.032 | [0.963, 1.100] |  |
| driver_age | 55-65 | 15133 | 0.952 | [0.865, 1.026] |  |
| driver_age | 65-75 | 7596 | 1.040 | [0.953, 1.129] |  |
| driver_age | <25 | 4611 | 0.998 | [0.910, 1.101] |  |
| driver_age | >=75 | 3298 | 1.031 | [0.887, 1.154] |  |
| exposure | 0.25-0.75 | 33707 | 1.019 | [0.978, 1.061] |  |
| exposure | 0.75-1 | 10280 | 1.084 | [0.994, 1.158] |  |
| exposure | <0.25 | 32737 | 0.989 | [0.913, 1.054] |  |
| exposure | >=1 | 25311 | 0.970 | [0.931, 1.009] |  |
| region | R11 | 10419 | 0.993 | [0.913, 1.061] |  |
| region | R24 | 24233 | 1.007 | [0.945, 1.067] |  |
| region | R52 | 5924 | 1.003 | [0.889, 1.094] |  |
| region | R53 | 6264 | 0.925 | [0.830, 0.994] |  |
| region | R82 | 12564 | 0.947 | [0.877, 1.023] |  |
| region | R91 | 5458 | 1.046 | [0.884, 1.177] |  |
| region | R93 | 11793 | 1.035 | [0.942, 1.107] |  |
| region | other | 25380 | 1.057 | [0.984, 1.112] |  |
| vehicle_age | 1-3 | 19804 | 0.992 | [0.911, 1.053] |  |
| vehicle_age | 10-15 | 22537 | 0.983 | [0.930, 1.048] |  |
| vehicle_age | 3-6 | 20063 | 0.974 | [0.926, 1.029] |  |
| vehicle_age | 6-10 | 19728 | 1.062 | [1.012, 1.119] |  |
| vehicle_age | <1 | 8559 | 1.052 | [0.989, 1.108] |  |
| vehicle_age | >=15 | 11344 | 0.954 | [0.864, 1.030] |  |

## Explainability

Describes the fitted model, not causal effects. Computed on a validation sample, not the holdout.

Permutation importance (increase in exposure-weighted deviance when a feature is shuffled):

| feature | deviance increase |
|---|---|
| Exposure | 0.205545 |
| BonusMalus | 0.037672 |
| VehAge | 0.024707 |
| DrivAge | 0.012538 |
| VehBrand | 0.012235 |
| VehPower | 0.004338 |
| VehGas | 0.003863 |
| Region | 0.003651 |
| Density | 0.001430 |
| Area | -0.000038 |

TreeSHAP via LightGBM pred_contrib (log-rate scale). Mean |SHAP| by feature:

| feature | mean abs SHAP |
|---|---|
| Exposure | 0.5290 |
| BonusMalus | 0.2088 |
| VehAge | 0.1422 |
| DrivAge | 0.1141 |
| Region | 0.0920 |
| VehBrand | 0.0525 |
| Density | 0.0513 |
| VehPower | 0.0452 |
| VehGas | 0.0335 |
| Area | 0.0060 |

Partial dependence, `DrivAge`: 22.0: 0.3299, 31.0: 0.2498, 38.0: 0.2620, 43.0: 0.2874, 48.0: 0.2937, 53.0: 0.2784, 60.0: 0.2666, 78.0: 0.2824

Partial dependence, `BonusMalus`: 50.0: 0.2436, 60.0: 0.2748, 76.0: 0.3002, 100.0: 0.4066

Partial dependence, `VehAge`: 0.0: 0.4210, 1.0: 0.2129, 3.0: 0.2151, 5.0: 0.2218, 7.0: 0.2215, 10.0: 0.2127, 13.0: 0.1871, 19.0: 0.1398

Partial dependence, `Density`: 12.0: 0.2512, 48.0: 0.2642, 114.0: 0.2636, 252.0: 0.2678, 539.15: 0.2719, 1313.0: 0.2726, 3317.0: 0.2706, 15703.0: 0.2648

## Limitations

- **No timestamps**: temporal generalization, seasonality and trend were not evaluated.
- **Simulated traffic and drift**: shadow/canary/drift results come from replaying held-out policies and from synthetic perturbations; they are demonstrations of the machinery, not evidence about real production.
- Frequency only; no severity. One country/product/period; extrapolation elsewhere is untested.
- Small slices are flagged `insufficient data` and not judged. Bootstrap CIs resample policies and ignore policy-to-policy dependence.
- The audit log is a local file: tamper-*evident*, not tamper-*proof*.

### Sensitive features and proxy risk

`DrivAge` and `Region` (and, indirectly, `Density`/`Area`) can act as proxies for protected characteristics (age, and in some jurisdictions ethnicity, religion or socio-economic status). They are kept for realism and reported by slice; this card does **not** claim the model is fair, compliant, or fit for pricing or underwriting.

## Provenance

| field | value |
|---|---|
| candidate hash | `d5ca3193399bdeea` |
| git SHA | `no-git-commit` |
| data version | `fremtpl2-162afa27d8` |
| seed | 20260101 |
| lockfile hash | `325e385df3aac6ed` |
| gate config hash | `9a4adf8a04377cef` |
