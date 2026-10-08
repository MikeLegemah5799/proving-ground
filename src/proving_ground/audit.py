"""Append-only, hash-chained audit log (JSONL).

Each entry's ``entry_hash`` is SHA-256 over the canonical serialization
(see ``canonical.py``) of the entry *without* its ``entry_hash`` field, and
``prev_entry_hash`` links to the previous entry (64 zeros for the first).

A local file is tamper-*evident*, not tamper-*proof*: anyone who can rewrite
the whole file can recompute the chain. Anchor the latest hash somewhere you
trust (a CI artifact, a signed commit) if that matters to you.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import canonical_hash

GENESIS = "0" * 64
EVENTS = ("promote", "reject", "hold", "rollback", "canary_abort")
FIELDS = (
    "seq", "ts", "event", "candidate_hash", "champion_hash_before", "champion_hash_after",
    "git_sha", "data_version", "gate_config_hash", "evidence", "reason", "actor", "synthetic",
    "prev_entry_hash",
)


def entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    return canonical_hash(body)


@dataclass
class VerifyResult:
    ok: bool
    entries: int
    error: str | None = None
    broken_at_seq: int | None = None
    head: str = GENESIS
    details: list[str] = field(default_factory=list)


def read_entries(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").split("\n"):  # NOT splitlines(): U+2028 etc. may appear inside JSON strings
        if line.strip():
            out.append(json.loads(line))
    return out


def verify_entries(entries: list[dict[str, Any]]) -> VerifyResult:
    prev = GENESIS
    for i, e in enumerate(entries):
        seq = e.get("seq")
        if seq != i + 1:
            return VerifyResult(False, len(entries), f"sequence gap or reorder: expected seq {i + 1}, found {seq}", i + 1, prev)
        if e.get("prev_entry_hash") != prev:
            return VerifyResult(False, len(entries), f"prev_entry_hash mismatch at seq {seq} (entry removed or reordered)", seq, prev)
        if e.get("entry_hash") != entry_hash(e):
            return VerifyResult(False, len(entries), f"entry_hash mismatch at seq {seq} (entry edited)", seq, prev)
        prev = e["entry_hash"]
    return VerifyResult(True, len(entries), None, None, prev)


def verify(path: str | Path) -> VerifyResult:
    try:
        entries = read_entries(path)
    except json.JSONDecodeError as exc:
        return VerifyResult(False, 0, f"unparseable line: {exc}", None)
    return verify_entries(entries)


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def entries(self) -> list[dict[str, Any]]:
        return read_entries(self.path)

    def verify(self) -> VerifyResult:
        return verify(self.path)

    def append(
        self,
        event: str,
        *,
        reason: str,
        candidate_hash: str | None = None,
        champion_hash_before: str | None = None,
        champion_hash_after: str | None = None,
        git_sha: str | None = None,
        data_version: str | None = None,
        gate_config_hash: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
        actor: str = "proving-ground",
        synthetic: bool = False,
        ts: str | None = None,
    ) -> dict[str, Any]:
        if event not in EVENTS:
            raise ValueError(f"unknown audit event {event!r}; expected one of {EVENTS}")
        res = self.verify()
        if not res.ok:
            raise RuntimeError(f"refusing to append to a broken audit chain: {res.error}")
        from .provenance import utc_now

        entry: dict[str, Any] = {
            "seq": res.entries + 1,
            "ts": ts or utc_now(),
            "event": event,
            "candidate_hash": candidate_hash,
            "champion_hash_before": champion_hash_before,
            "champion_hash_after": champion_hash_after,
            "git_sha": git_sha,
            "data_version": data_version,
            "gate_config_hash": gate_config_hash,
            "evidence": evidence or [],
            "reason": reason,
            "actor": actor,
            "synthetic": synthetic,
            "prev_entry_hash": res.head,
        }
        entry["entry_hash"] = entry_hash(entry)  # also rejects floats
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        return entry
