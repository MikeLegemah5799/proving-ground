# Adopting Proving Ground: the five touchpoints

Everything else (gates, registry, shadow/canary, drift, audit, viewer) is reused unchanged.
The toy project in [examples/toy/](examples/toy/) is the smallest complete example (about 90 lines).

## 1. Implement `ModelProject`

[src/proving_ground/interface.py](src/proving_ground/interface.py). The eight contract methods from the spec plus four small
extensions the gates need:

| Method | Purpose |
|---|---|
| `load_data(data_version) -> DataBundle` | train / val / holdout splits, a **reference** sample (never in train), optional **stream** (replayed as simulated traffic). Splits must not share `id_col` values |
| `schema()` | Pandera schema of the raw input (used for G1 **and** request validation in serving) |
| `train(bundle, seed, params)` | return any picklable model; must be deterministic for a given seed |
| `predict(model, X)` | one prediction per row |
| `metrics(model, bundle)` | primary metrics on the holdout |
| `slices()` | `SliceSpec`s for G5 |
| `calibration_report(model, bundle)` | domain calibration scalars (G4) |
| `monitored_features()` | columns to monitor for drift |
| *ext.* `score(y, pred, weight)` | array-level metrics, used for bootstrap CIs, slices and label-drift monitoring |
| *ext.* `metric_directions()` | `"higher"` or `"lower"` is better, per metric (G3) |
| *ext.* `render_model_card(ctx)` | markdown for G8, regenerated for every candidate |
| *ext.* `default_params(family)` | hyperparameters per model family |

Optional: `drift_scenarios()` (domain scenarios), `calibration_arrays(y, pred, w)` (decile table for the viewer), `tune(...)`.

## 2. `config/project.yaml`
Set `project: your_package.module:YourProject` and paths. The `demo:` block configures the scripted story.

## 3. Data under DVC
Edit `dvc.yaml` so `fetch`/`prepare` produce your data. Do not commit raw data you cannot redistribute.

## 4. `config/gates.yaml`
Set thresholds **after** training a baseline and a first model. Record how you chose each number (see
[examples/claims_frequency/docs/GATES.md](examples/claims_frequency/docs/GATES.md)). The hash of this file is stored in every gate report;
changing it invalidates earlier reports (promotion re-checks).

## 5. Model card template
Edit `render_model_card`. G8 requires the sections listed in `g8_card.required_sections` and the candidate hash in the text.

## Check your work
```bash
make test        # core tests never depend on your project
make demo-fast   # or point config/project.yaml at your project and run `proving-ground demo`
```
