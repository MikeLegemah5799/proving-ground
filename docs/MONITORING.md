# Monitoring and drift scenarios

Windows of logged requests (`logs/requests.jsonl`) are compared with the reference sample (a sample of the validation split).
Per feature: PSI and KS (numeric), PSI and chi-squared (categorical), **new-category rate** and **out-of-range rate**;
plus prediction PSI, serving **validation-failure rate**, and, when delayed labels arrive, **label drift** (`ae_ratio` against the reference value).
Levels `ok < watch < alert` come from `config/monitoring.yaml`; reference predictions use the champion that actually served the window.
Each window writes `window_NN.json` (schema-versioned, `synthetic` flag) and, if Evidently is installed, `window_NN.html`.
An `alert` writes a `hold` audit entry with a rollback recommendation.

## Decisions that came from evidence
- **Label drift must be statistically significant.** A first version alerted whenever A/E moved by 20%. With ~130 claims per window, noise alone moves A/E by about 10%, and `silent_decay` raised an early false alert (no `watch` first). The monitor now requires the bootstrap CI (99%) of the window metric to exclude the reference value. Cost: it cannot see small base-rate changes in small windows (below).
- PSI on tiny samples is biased upward; the canary guardrail uses fewer bins when few canary rows exist.

## Scenario matrix (all `synthetic: true`)
`make demo` runs every scenario against the current champion in a scratch log directory (the real logs are untouched) and records declared vs observed in `reports/drift/scenario_matrix.json`. Window levels below are for 2,500-request windows on the real data (first value = clean baseline window).

| Scenario | Declared expectation | Observed levels |
|---|---|---|
| `young_driver_surge` | alert on DrivAge | ok ok ok **watch alert** alert alert |
| `silent_decay` | alert on VehAge, **watch before alert** | ok ok ok ok **watch watch watch watch alert** alert |
| `new_vehicle_brand` | alert on VehBrand (new-category rate; serving also rejects the unseen brand) | ok ok alert … |
| `bonus_malus_schema_break` | alert on validation failures | ok ok alert … |
| `claims_frequency_shift` | alert on label drift | ok ok ok **alert** alert alert |
| generic: `covariate_shift`, `new_category`, `schema_break`, `label_shift` | see `injection.py` | all detected |

Every declared expectation was met in the recorded run. That is **not** a claim of general sensitivity:

## What the monitor cannot see (limits, stated up front)
- **Small label shifts in small windows.** `label_shift` / `claims_frequency_shift` ramp up to +60% / +50% extra claims. They were *not* detected at +15–30% (`label_shift`) and +12–25% (`claims_frequency_shift`) (windows of 2,500 requests contain ~130 claims). Use larger windows or lower-variance statistics for real use.
- **Timing is simulated.** Windows are request batches, not calendar time; "slow" drift is a ramp over windows. Nothing was tested for seasonality or real feedback delay.
- **Thresholds were set by the author** from this dataset's reference noise, not tuned against real incidents.
- **After rollback the drift is still there** (the data did not change); levels stay `alert`. The viewer says so.
- A `new_vehicle_brand` is rejected by the strict schema, so users get a validation error for those policies rather than a prediction. Whether to reject or score-with-a-fallback is a product decision this template does not make.
