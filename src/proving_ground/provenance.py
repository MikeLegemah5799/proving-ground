"""Provenance: git SHA, lockfile hash, config hashes, timestamps."""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .canonical import canonical_hash, sha256_hex


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha(root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, cwd=root, timeout=10
        )
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "no-git-commit"
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=root, timeout=10)
        return sha + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "no-git"


def lockfile_hash(root: Path) -> str:
    for name in ("requirements.lock", "uv.lock", "poetry.lock"):
        p = root / name
        if p.exists():
            return sha256_hex(p.read_bytes())[:16]
    return "no-lockfile"


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def config_hash(path: str | Path) -> str:
    """Hash of the parsed YAML content (insensitive to comments/whitespace)."""
    return canonical_hash(load_yaml(path), allow_floats=True)[:16]


def provenance(root: Path, data_version: str, seed: int, gate_config: Path) -> dict[str, str | int]:
    return {
        "git_sha": git_sha(root),
        "data_version": data_version,
        "seed": seed,
        "lockfile_hash": lockfile_hash(root),
        "gate_config_hash": config_hash(gate_config),
    }
