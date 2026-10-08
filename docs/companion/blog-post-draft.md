# What broke, and how the pipeline caught it

*Draft for mleg.tech. Written for: engineers evaluating how a model release should be gated. Numbers are from a real `make demo` run on freMTPL2freq (see README for the one-time download); all traffic and drift are simulated and labeled `synthetic: true`. Match the style of the site's existing posts and replace the placeholder links before publishing.*

## The setup
A motor-insurance claim-frequency model, released through Proving Ground: gates G1–G8, shadow, a 10% canary, drift monitoring, a hash-chained audit log, and a rollback that has been exercised in CI.
Three models, one grouped holdout of 102,035 policies, exposure-weighted metrics with bootstrap CIs:

| | Poisson deviance | Gini |
|---|---|---|
| Constant baseline | 0.6237 [0.610, 0.636] | 0 |
| Poisson GLM | 0.5774 [0.566, 0.589] | 0.328 |
| LightGBM | 0.5528 [0.541, 0.563] | 0.378 |

## Break #1: the data lied about exposure (caught by a slice gate)
The first GLM looked fine on aggregate. The exposure-band slice did not: actual/expected claims were **2.26 for policies under a quarter-year of exposure and 0.84 for full-year ones**. Claims aren't proportional to exposure in this dataset.
Entering exposure as a covariate fixed it (and lifted both models). The finding is in FEATURES.md, with the caveat that this is wrong for pricing.

## Break #2: a bad challenger (rejected by named gates)
A challenger with two columns swapped in preprocessing was trained and gated. It failed **G2, G3, G4 and G5**; the rejection is audit entry #2, and the registration command refuses it.

## Break #3: the drift incident
After LightGBM was promoted (shadow: Spearman 0.90 vs the GLM; canary: 489 canary / 4,511 control requests; promoted as audit #3), synthetic drift began: a younger cohort over-represented in traffic (`young_driver_surge`), ramped over six windows.
- Windows 4–7: `ok`. Window 8 (intensity 0.09): **watch**: DrivAge PSI crossed 0.10. Window 9 (0.20): **alert**.
- Audit #4 (`hold`): "Drift alert in window 9 (drivers: DrivAge). Rollback to previous champion 13f8c79b… recommended."
- Audit #5 (`rollback`): GBM → GLM, restored hash verified via `/health`; a second `rollback` call changed nothing.
- The audit chain verified end to end; the report viewer reproduces the story from the log and drift reports.

**Be careful what you conclude.** Calibration (A/E) stayed within noise throughout (0.88–1.13 on 2,500-row windows), and the GLM after rollback was *not* better than the GBM on deviance under the same drift (0.78–0.80 vs 0.75 at full intensity): the deviance rise came from the population mix, not from the model failing.
Rollback restored a known-good state; it did not fix the data. The monitor did its job (it saw the inputs leave the validated envelope); whether rolling back was the *right* response is a judgment the runbook leaves to a human unless `auto_rollback` is on.

## What the monitor can't see
Label shifts of +15–30% went undetected in 2,500-request windows (about 130 claims); a first version that alerted on raw A/E changes false-alarmed on `silent_decay` until label drift was required to be statistically significant. See docs/MONITORING.md.

## Try it
`make setup && make data && make demo` (1.4 minutes measured), then open `reports/viewer/index.html`. Live sample: *[link after the first Pages deploy]*. Template: *[GitHub link]*.
