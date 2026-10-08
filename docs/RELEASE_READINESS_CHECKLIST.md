# Model Release Readiness Checklist

Derived from the Proving Ground gate suite. Tick every box before a model goes to real traffic. "Evidence" is something another person can open and check.

## Data (G1)
- [ ] Schema of raw inputs is written down and enforced at training **and** serving time
- [ ] Null rates, category sets and ranges are checked; row floors are set
- [ ] No identifier (or near-duplicate record) appears in more than one split
- [ ] Data provenance recorded: source, license, download date, checksum; raw data never edited in place
- [ ] You state what the data cannot show (for example: no timestamps, so no temporal validation)

## Evaluation (G2–G5)
- [ ] Metrics use the right weights (exposure, sample weights) and are checked on a constructed case
- [ ] Absolute floors exist and were set *after* seeing baseline results, with the reasoning recorded
- [ ] Compared with the current champion on the same holdout, with a confidence interval
- [ ] The holdout was evaluated once for this candidate; tuning used cross-validation inside training only
- [ ] Calibration checked with a number that gates, not just a chart
- [ ] Slices declared (including ones that can proxy for protected characteristics); small slices reported as "insufficient data", never silently passed

## Engineering (G6–G8)
- [ ] Latency and artifact size budgets measured on the serving wrapper
- [ ] Same seed and data give the same candidate hash (or a declared epsilon)
- [ ] Model card regenerated for *this* candidate: intended and out-of-scope use, data, metrics with CIs, limitations

## Release
- [ ] Promotion is impossible without a passing gate report for this exact candidate (no override flag)
- [ ] Shadow comparison passed (agreement, rank correlation, latency, exceptions)
- [ ] Canary with stable routing and automatic abort on guardrail breach
- [ ] Rollback restores the previous champion without redeploying, and is idempotent

## Operations
- [ ] Rollback was **drilled** in CI, not just written down
- [ ] Drift monitors cover features, new categories, out-of-range values, predictions, validation failures and (when available) labels
- [ ] You know what your monitors *cannot* see (window size, label delay, seasonality)
- [ ] Every decision lands in an append-only, hash-chained log; the head hash is anchored somewhere you trust
- [ ] Synthetic and simulated data are labeled wherever they appear
