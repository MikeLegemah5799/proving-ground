# How the claims thresholds were chosen

Rule followed (spec §9.4): **no thresholds were invented up front.** The constant baseline and the Poisson GLM were trained and evaluated first
(holdout of 102,035 policies, exposure-weighted, 200 bootstrap resamples), then each number below was set from what was observed.
Thresholds live in [config/gates.yaml](../../../config/gates.yaml); `config/gates.synthetic.yaml` holds looser **smoke-test** thresholds for the 40k-row synthetic stand-in.

| Observed (holdout) | Constant | GLM | LightGBM |
|---|---|---|---|
| Poisson deviance (per exposure-year) | 0.6237 [0.610, 0.636] | 0.5774 [0.566, 0.589] | 0.5528 [0.541, 0.563] |
| Deviance skill vs constant | 0 | 0.0742 | 0.1138 |
| Gini | 0 | 0.328 [0.311, 0.343] | 0.378 [0.362, 0.393] |
| Decile calibration error | 0.496 | 0.061 | 0.043 |

| Gate / key | Value | How it was chosen |
|---|---|---|
| G2 `deviance_skill` min | 0.04 | Well below the GLM's observed 0.074: a model must clearly beat the constant, and a plain GLM must pass with room for noise |
| G2 `gini` min | 0.25 | Below the GLM's lower CI bound (0.311) by about 0.06; excludes a model that barely ranks risk |
| G2 `poisson_deviance` max | 0.600 | Between the constant (0.624, CI lower 0.610) and the GLM's CI upper bound (0.589) |
| G3 deviance, relative worsening | 0.5% | About the width of the paired-bootstrap noise for two models of this size; the point estimate must not be worse by more |
| G3 gini, relative worsening | 2% | Gini is noisier than deviance (its bootstrap CI is about ±0.015 on a value of 0.33, i.e. ±5%), so it gets a looser tolerance |
| G4 `decile_calibration_error` max | 0.10 | The best models show 0.04–0.06, which is mostly sampling noise at ~10k exposure per decile; the constant shows 0.50. 0.10 admits noise and rejects real miscalibration |
| G5 `ae_ratio` gap | 0.20 | Observed worst slice gap for the GLM and LightGBM is under 0.09 with n ≥ 2000; the 95% CI half-width of a 2,000–5,000-policy slice is roughly 0.08–0.15. A point-estimate gap over 0.20 is a real problem, a CI that crosses 0.20 only warns |
| G5 `min_samples` | 2000 policies | ≈100 claims; below that a slice cannot be judged and is reported as `insufficient data` |
| G6 p95 latency | 30 ms | GLM measured 3.2 ms single-row in-process; ~10× headroom |
| G6 artifact | 50 MB | Both models are well under 1 MB |
| G7 epsilon | 1e-6 | Only used if a retrain hash differs (it did not) |

## What the gates caught (real runs)
- **Exposure** (before the fix): the exposure-band slice failed badly for the first GLM: A/E 2.26 for exposure < 0.25, 0.84 for ≥ 1. See FEATURES.md.
- **Degraded challenger** (two columns swapped in preprocessing): rejected with G2 (floors), G3 (regression), G4 (calibration), G5 (slices).
- **bonus_malus ≥ 110** has 888 holdout policies, below `min_samples`, so every candidate carries a G5 warning for it: *insufficient data*, by design not a pass.

## Caveats
Thresholds were fitted to one random grouped split of one dataset. They are a worked example of the process, not recommendations.
