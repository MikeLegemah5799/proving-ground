# Gate suite

All gates run in order and all must pass. Results are `pass | fail | warn | skip`. A `warn` never blocks but is shown everywhere. Output: JSON + markdown per candidate under `reports/<hash>/`.

| Gate | Checks | Config key |
|---|---|---|
| G1 Data validation | Pandera schema per split, null rate, row floors, **no id in more than one split** | `g1_data` |
| G2 Metric floors | absolute min/max on holdout metrics | `g2_floors` |
| G3 No regression | vs champion on the **same holdout**, paired bootstrap; fail if the *point estimate* worsens beyond tolerance, warn if the CI upper bound does. First promotion: `skip`, recorded | `g3_regression` |
| G4 Calibration | domain calibration error within bound | `g4_calibration` |
| G5 Slices | each slice within `max_gap` of the target; slices below `min_samples` are `warn: insufficient data` (never a silent pass); warn when the CI exceeds the bound | `g5_slices` |
| G6 Performance budget | p95 single-row `predict()` latency, pickle size | `g6_budget` |
| G7 Reproducibility | retrain with the same seed: same hash, or metrics within epsilon | `g7_repro` |
| G8 Model card | exists, names this candidate hash, contains required sections | `g8_card` |

## Holdout discipline
`evaluation.py` scores the holdout once per candidate and caches predictions. Re-running gates after a threshold change reuses the cache.
`reports/holdout_ledger.jsonl` has one line per candidate (tested). Tuning uses cross-validation inside the training split only.

## Changing thresholds
Edit `config/gates.yaml`. Its content hash is stored in every report and re-checked at registration and promotion, so older reports
stop being valid and gates must re-run. Document *why* you chose each number (see the claims example's GATES.md).
