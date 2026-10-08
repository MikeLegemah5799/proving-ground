"""File-based model registry with aliases and a release-stage state machine.

Layout::

    registry/
      models/<hash>/model.pkl, meta.json
      aliases.json     {"candidate": h|null, "champion": h|null, "previous_champion": h|null}
      state.json       {"stage": ..., "canary_pct": int, "rolled_back_from": h|null}

Models are pickled: only load a registry you created yourself (see SECURITY.md).
The alias flip is the only thing a rollback needs. ``tracking.py`` can mirror runs and aliases
into MLflow (opt-in).
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

from .candidate import Candidate

ALIASES = ("candidate", "champion", "previous_champion")
STAGES = ("none", "registered", "shadow", "shadow_passed", "canary", "canary_passed", "stable", "aborted")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # --- models ---------------------------------------------------------------
    def model_dir(self, h: str) -> Path:
        return self.root / "models" / h

    def save(self, cand: Candidate, extra_meta: dict[str, Any] | None = None) -> Path:
        d = self.model_dir(cand.hash)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "model.pkl").open("wb") as fh:
            pickle.dump(cand, fh, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            "hash": cand.hash, "project": cand.project, "family": cand.family, "seed": cand.seed,
            "data_version": cand.data_version, "params": cand.params, **(extra_meta or {}),
        }
        _atomic_write(d / "meta.json", json.dumps(meta, indent=2, sort_keys=True, default=str))
        return d

    def exists(self, h: str) -> bool:
        return (self.model_dir(h) / "model.pkl").exists()

    def load(self, h: str) -> Candidate:
        with (self.model_dir(h) / "model.pkl").open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - local, self-produced artifacts only

    def meta(self, h: str) -> dict[str, Any]:
        return json.loads((self.model_dir(h) / "meta.json").read_text())

    # --- aliases ----------------------------------------------------------------
    def aliases(self) -> dict[str, str | None]:
        p = self.root / "aliases.json"
        base = {a: None for a in ALIASES}
        if p.exists():
            base.update(json.loads(p.read_text()))
        return base

    def get_alias(self, alias: str) -> str | None:
        return self.aliases()[alias]

    def set_aliases(self, **updates: str | None) -> dict[str, str | None]:
        for k in updates:
            if k not in ALIASES:
                raise KeyError(k)
        cur = self.aliases()
        cur.update(updates)
        _atomic_write(self.root / "aliases.json", json.dumps(cur, indent=2, sort_keys=True))
        return cur

    # --- stage state ------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        p = self.root / "state.json"
        base = {"stage": "none", "canary_pct": 0, "rolled_back_from": None}
        if p.exists():
            base.update(json.loads(p.read_text()))
        return base

    def set_state(self, **updates: Any) -> dict[str, Any]:
        st = self.state()
        if "stage" in updates and updates["stage"] not in STAGES:
            raise ValueError(f"unknown stage {updates['stage']}")
        st.update(updates)
        _atomic_write(self.root / "state.json", json.dumps(st, indent=2, sort_keys=True))
        return st
