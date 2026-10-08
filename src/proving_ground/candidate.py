"""A trained candidate plus its deterministic identity hash.

Hash definition (documented in docs/ARCHITECTURE.md): SHA-256 over the canonical
serialization of {project, family, params, seed, data_version, fingerprint} where
``fingerprint`` is a SHA-256 of the candidate's predictions, rounded to 6 decimals,
on the first 1000 rows of the bundle's reference sample. Pickle bytes are *not*
hashed because they are not stable across library versions.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .canonical import canonical_hash
from .interface import DataBundle, ModelProject


@dataclass
class Candidate:
    model: Any
    project: str
    family: str
    params: dict
    seed: int
    data_version: str
    hash: str = ""
    notes: dict[str, Any] = field(default_factory=dict)


def fingerprint(project: ModelProject, model: Any, bundle: DataBundle, n: int = 1000) -> str:
    X = bundle.reference[bundle.features].head(n)
    pred = np.asarray(project.predict(model, X), dtype=float)
    txt = ",".join(f"{v:.6f}" for v in pred)
    return hashlib.sha256(txt.encode()).hexdigest()


def build_candidate(
    project: ModelProject, bundle: DataBundle, *, family: str, seed: int, params: dict | None = None
) -> Candidate:
    params = {**project.default_params(family), **(params or {})}
    t0 = time.perf_counter()
    model = project.train(bundle, seed, {**params, "_family": family})
    elapsed = time.perf_counter() - t0
    ident = {
        "project": project.name,
        "family": family,
        "params": {k: (round(v, 10) if isinstance(v, float) else v) for k, v in sorted(params.items())},
        "seed": seed,
        "data_version": bundle.data_version,
        "fingerprint": fingerprint(project, model, bundle),
    }
    h = canonical_hash(ident, allow_floats=True)[:16]
    return Candidate(model, project.name, family, params, seed, bundle.data_version, h, {"train_seconds": round(elapsed, 2)})
