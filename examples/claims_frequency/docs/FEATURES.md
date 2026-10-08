# Features, caps and bins

Raw inputs (the serving API accepts exactly these): `Exposure, Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region`. `IDpol` is only a key.
Caps are applied **inside the model pipeline**, never in the stored data, so serving applies exactly what training did.

| Feature | Why it is there | GLM treatment | LightGBM treatment |
|---|---|---|---|
| `DrivAge` | strongest classical rating factor (young and very old drivers differ) | cap 90, bins at 18,21,25,30,…,80 → one-hot | numeric |
| `BonusMalus` | prior claims history (50 = best) | cap 150, bins 50,51,60,70,80,95,110,130 → one-hot | numeric |
| `VehAge` | older vehicles: different usage and claim patterns | cap 20, bins 0,1,2,3,5,8,11,15,20 → one-hot | numeric |
| `VehPower` | engine power | cap 12, bins 4…11 → one-hot | numeric |
| `Density` | urban vs rural exposure to traffic | log1p, standardized | numeric |
| `Area` | density-derived area code A–F | one-hot | categorical |
| `VehBrand`, `VehGas`, `Region` | rating factors in the source data | one-hot, unseen level → all-zero (baseline) | categorical, unseen level → missing |
| `Exposure` | see below | bins at 0.1,0.25,0.5,0.75,0.99 → one-hot | numeric |

Also used as the **sample weight** and in the Poisson loss: the model predicts a rate (claims per exposure-year) and expected claims = rate × exposure.

## The exposure finding
Data cleaning caps `Exposure` at 1 year and `ClaimNb` at 4. Even so, claims are **not proportional to exposure** in freMTPL2freq: with exposure used only
as a weight, the first GLM had A/E (actual/expected claims) of **2.26 for exposure < 0.25** and **0.84 for exposure ≥ 1** on the holdout (LightGBM: 2.22 / 0.82).
Short policies show much higher frequency per exposure-year (a known property of this data; a plausible but unverified reason is that short terms are
disproportionately policies ended by a claim). Entering exposure as a covariate removed the bias (all slice gaps then under 0.09) and improved both models
(GLM deviance skill 0.051 → 0.074, LightGBM 0.079 → 0.114, same split).
**Caveat:** this is a pragmatic fix. It means the model's rate depends on the policy's observed exposure, which would not be known when quoting a new annual policy. It
is acceptable for this demonstration of release machinery and is wrong for pricing.

## Sensitive features
`DrivAge`, `Region` (and via correlation `Density`/`Area`) can proxy for protected characteristics. They are kept, reported by slice (G5), and the proxy risk is stated in the model card. No fairness claim is made.
