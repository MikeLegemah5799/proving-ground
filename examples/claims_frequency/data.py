"""freMTPL2freq: fetch, checksum, clean, split. Raw data is never edited in place.

    python -m examples.claims_frequency.data fetch      # download raw ARFF + provenance record
    python -m examples.claims_frequency.data prepare    # clean -> data/clean/*.parquet + quality report
    python -m examples.claims_frequency.data synthetic  # generate the labeled-synthetic stand-in

The dataset is NOT committed to this repository (see docs/DATA_CARD.md for license notes).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR, CLEAN_DIR = ROOT / "data" / "raw", ROOT / "data" / "clean"
RAW_FILE = RAW_DIR / "freMTPL2freq.arff"
CLEAN_FILE, SYN_FILE = CLEAN_DIR / "freMTPL2freq_clean.parquet", CLEAN_DIR / "synthetic_clean.parquet"
URL = "https://www.openml.org/data/v1/download/20649148/freMTPL2freq.arff"
EXPECTED_MD5 = "f8875568bf0ca622929105197e2db613"   # OpenML-published checksum (data id 41214)
COLUMNS = ["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]
PROFILE = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]
FEATURES = ["Exposure", *PROFILE]
CLAIM_CAP, EXPOSURE_CAP = 4, 1.0


def _hash_file(p: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        print(f"raw file present: {RAW_FILE}")
    else:
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, RAW_FILE)
    md5, sha = _hash_file(RAW_FILE, "md5"), _hash_file(RAW_FILE, "sha256")
    if md5 != EXPECTED_MD5:
        raise SystemExit(f"checksum mismatch: got md5 {md5}, expected {EXPECTED_MD5}. Do not use this file.")
    (RAW_DIR / "PROVENANCE.json").write_text(json.dumps({
        "dataset": "freMTPL2freq", "source_url": URL, "openml_data_id": 41214,
        "license_as_listed_on_openml": "CC0",
        "license_note": "CASdatasets (the original distribution) is GPL-licensed; the OpenML listing says CC0. "
                        "This repo does not redistribute the data: it is fetched at setup and kept out of git.",
        "downloaded": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"), "bytes": RAW_FILE.stat().st_size,
        "md5": md5, "sha256": sha,
    }, indent=2))
    print(f"ok: md5 {md5} matches OpenML")
    return RAW_FILE


def read_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    skip = 0
    with path.open() as fh:
        for i, line in enumerate(fh):
            if line.lower().startswith("@data"):
                skip = i + 1
                break
    df = pd.read_csv(path, skiprows=skip, names=COLUMNS, quotechar="'", low_memory=False)
    df["IDpol"] = df["IDpol"].astype("int64")
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Documented rules; every rule reports how many rows it touched. No rows are dropped."""
    df = raw.copy()
    rules = []

    def rule(name, mask, action):
        n = int(mask.sum())
        rules.append({"rule": name, "rows_affected": n, "share": round(n / len(raw), 6), "action": action})

    rule("ClaimNb > 4", df.ClaimNb > CLAIM_CAP, f"capped at {CLAIM_CAP} (extreme counts distort Poisson fits)")
    df["ClaimNb"] = df.ClaimNb.clip(upper=CLAIM_CAP)
    rule("Exposure > 1", df.Exposure > EXPOSURE_CAP, f"capped at {EXPOSURE_CAP} year (one-year policy terms)")
    df["Exposure"] = df.Exposure.clip(upper=EXPOSURE_CAP)
    rule("Exposure < 0.01 (< ~4 days)", df.Exposure < 0.01, "kept and flagged only; rates on tiny exposure are very noisy")
    rule("VehAge > 40", df.VehAge > 40, "kept; capped inside the model pipeline, not in the data")
    rule("BonusMalus > 150", df.BonusMalus > 150, "kept; capped inside the model pipeline, not in the data")
    rule("duplicate IDpol", df.IDpol.duplicated(), "none expected; checked")
    for c in ("Area", "VehBrand", "VehGas", "Region"):
        df[c] = df[c].astype(str)
    prof = df.groupby(PROFILE).ngroups
    report = {
        "rows_raw": len(raw), "rows_clean": len(df), "null_cells": int(raw.isna().sum().sum()), "rules": rules,
        "claim_rate_per_exposure_year": float(df.ClaimNb.sum() / df.Exposure.sum()), "total_exposure_years": float(df.Exposure.sum()),
        "share_policies_with_claim": float((df.ClaimNb > 0).mean()), "distinct_risk_profiles": int(prof),
        "note": "Rows sharing an identical risk profile (all rating factors equal) are kept in the same split to avoid near-duplicate leakage.",
    }
    return df, report


def data_version_of(df: pd.DataFrame, prefix: str) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:10]
    return f"{prefix}-{h}"


def prepare() -> Path:
    if not RAW_FILE.exists():
        raise SystemExit("raw data missing: run `make data` (fetch) first")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df, report = clean(read_raw())
    report["data_version"] = data_version_of(df, "fremtpl2")
    df.to_parquet(CLEAN_FILE, index=False)
    (CLEAN_DIR / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    print(f"clean data: {len(df)} rows -> {CLEAN_FILE} ({report['data_version']})")
    for r in report["rules"]:
        print(f"  {r['rule']:<28} {r['rows_affected']:>7} rows  {r['action']}")
    return CLEAN_FILE


def synthetic(n: int = 60000, seed: int = 0) -> pd.DataFrame:
    """A freMTPL2-like stand-in with a known data-generating process. Labeled synthetic everywhere it is used.

    Exists so CI and offline fresh clones never depend on the network. It is NOT real insurance data."""
    rng = np.random.default_rng(seed)
    drivage = np.clip(rng.normal(45, 15, n).astype(int), 18, 99)
    vehage = np.clip(rng.gamma(2, 3.5, n).astype(int), 0, 40)
    area = rng.choice(list("ABCDEF"), n, p=[.1, .15, .2, .25, .2, .1])
    dens = np.exp(rng.normal(5.5, 1.8, n)).astype(int).clip(1, 27000)
    bm = np.clip((50 + rng.gamma(1.2, 8, n) * (drivage < 30) + rng.gamma(1.0, 9, n)).astype(int), 50, 200)
    brand = rng.choice([f"B{i}" for i in (1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14)], n)
    gas = rng.choice(["Regular", "Diesel"], n)
    region = rng.choice([f"R{i}" for i in (11, 21, 22, 23, 24, 25, 26, 31, 41, 42, 43, 52, 53, 54, 72, 73, 74, 82, 83, 91, 93, 94)], n)
    power = rng.integers(4, 13, n)
    expo = np.clip(np.where(rng.random(n) < 0.35, rng.uniform(0.01, 1, n), 1.0) * rng.uniform(0.5, 1, n), 0.003, 1)
    log_rate = (-2.6 + 1.3 * (drivage < 25) + 0.5 * (drivage < 35) - 0.35 * (drivage > 55) + 0.03 * (bm - 50) + 0.25 * np.log1p(dens) / 3
                - 0.08 * np.minimum(vehage, 15) / 3 + 0.4 * (gas == "Diesel") * (drivage < 35) + 0.25 * (power > 9)
                + 0.5 * (area == "F") * (vehage < 3))   # a mild interaction so tree models can beat the GLM
    claims = rng.poisson(np.exp(log_rate) * expo).clip(max=CLAIM_CAP)
    return pd.DataFrame({"IDpol": np.arange(1, n + 1), "ClaimNb": claims, "Exposure": expo.round(4), "Area": area,
                         "VehPower": power, "VehAge": vehage, "DrivAge": drivage, "BonusMalus": bm, "VehBrand": brand,
                         "VehGas": gas, "Density": dens, "Region": region})


def make_synthetic_file(n: int = 60000) -> Path:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df = synthetic(n)
    df.to_parquet(SYN_FILE, index=False)
    print(f"synthetic stand-in written: {SYN_FILE} ({len(df)} rows). NOT real data.")
    return SYN_FILE


def split_groups(df: pd.DataFrame, seed: int, proportions=(0.60, 0.10, 0.15, 0.15)) -> dict[str, pd.DataFrame]:
    """Policy- and risk-profile-grouped, claim-stratified split into train/val/holdout/stream.

    Rows sharing an identical risk profile (all rating factors equal; frequently the same vehicle across
    consecutive policy records) always land in the same split, so near-duplicates cannot leak across splits.
    Groups are shuffled and cut at the target proportions separately for claim / no-claim groups."""
    key = pd.util.hash_pandas_object(df[PROFILE], index=False).to_numpy()
    gids, inv = np.unique(key, return_inverse=True)
    has_claim = np.bincount(inv, weights=(df.ClaimNb.to_numpy() > 0).astype(float)) > 0
    rng = np.random.default_rng(seed)
    which_group = np.zeros(len(gids), dtype=int)
    cuts = np.cumsum(proportions)[:-1]
    for stratum in (True, False):
        idx = np.flatnonzero(has_claim == stratum)
        pos = rng.permutation(len(idx)) / max(len(idx), 1)
        which_group[idx] = np.searchsorted(cuts, pos, side="right")
    which = which_group[inv]
    names = ["train", "val", "holdout", "stream"]
    return {n: df[which == i].reset_index(drop=True) for i, n in enumerate(names)}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"fetch": fetch, "prepare": prepare, "synthetic": make_synthetic_file}[cmd]()
