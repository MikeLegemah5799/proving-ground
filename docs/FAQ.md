# FAQ

**Is there a `--force`?** No. Change the gates (and justify it in review) or fix the model.

**Why a file registry instead of MLflow?** Rollback is an alias flip, the demo runs offline in a minute, and there is nothing to host. `tracking: mlflow` mirrors runs and aliases into local MLflow.

**Why does the dataset need a download?** Redistribution rights are unclear (OpenML says CC0, the original CASdatasets distribution is GPL), so it is fetched at setup and never committed. `make data-synthetic` needs no network (labeled synthetic; used by CI).

**Is the audit log tamper-proof?** No, tamper-evident. See docs/ARCHITECTURE.md.

**Can I trust the shadow/canary results?** They prove the machinery works on simulated traffic. They say nothing about real production behavior.

**Why is `Exposure` a model feature?** In freMTPL2freq, claims are not proportional to exposure (short policies show far higher frequency per exposure-year). Before adding it, the A/E ratio was 2.26 for exposure < 0.25 and 0.84 for exposure ≥ 1. Details: examples/claims_frequency/docs/FEATURES.md.

**Does the report viewer need a server?** No. Open `reports/viewer/index.html` from disk. It makes no network requests.

**Why Python 3.14 in development but 3.11+ supported?** The newest compatible versions were used for development; CI exercises the oldest and newest supported ones.
