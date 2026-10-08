# Data card: freMTPL2freq

| | |
|---|---|
| Content | French motor third-party-liability policies: claim count, exposure (years), 9 rating factors |
| Source | OpenML data id 41214, `https://www.openml.org/data/v1/download/20649148/freMTPL2freq.arff` (36,288,567 bytes) |
| Original distribution | CASdatasets (Dutang et al.), associated with *Computational Actuarial Science with R* (Charpentier, ed., CRC 2018) |
| License | OpenML lists **CC0**; CASdatasets is distributed under **GPL**. Treat redistribution as unsettled: the file is **fetched at setup, never committed** |
| Integrity | md5 `f8875568bf0ca622929105197e2db613` (published by OpenML), verified on every fetch; sha256 and download date recorded in `data/raw/PROVENANCE.json` |
| Size | 678,013 rows (the OpenML description says 677,991; the file itself has 678,013 and is what is used), claim rate 0.1006 per exposure-year, 5.0% of policies with at least one claim |
| Raw vs clean | raw ARFF is untouched; `data/clean/*.parquet` is versioned separately (`data_version = fremtpl2-<hash of cleaned content>`) |

## Cleaning (row counts per rule: `data/clean/data_quality_report.json`)
`ClaimNb > 4` capped (9 rows) · `Exposure > 1` capped (1,224) · `Exposure < 0.01` flagged only (6,877) · `VehAge > 40` (230) and `BonusMalus > 150` (209) kept, capped inside the model pipeline · duplicate `IDpol`: 0. No rows are dropped.

## Known gaps (also in the model card)
- **No timestamps.** A time-based split is impossible and was not faked. Splits are **policy- and risk-profile-grouped, claim-stratified** (60/10/15/15 train/val/holdout/stream). Rows with identical rating factors (528,765 distinct profiles for 678,013 rows: about 22% of rows repeat another row's profile, often the same vehicle across policy records) always land in the same split. **Temporal generalization was not evaluated.**
- One country, one product, unknown period. No severity data used.
- Claims are not proportional to exposure (see FEATURES.md).
- Region and driver age may proxy for protected characteristics.

## Synthetic stand-in
`python -m examples.claims_frequency.data synthetic` generates a freMTPL2-like dataset with a known data-generating process. It exists so CI and offline clones never need the network. It is **labeled synthetic** everywhere it is used and says nothing about real insurance data.
