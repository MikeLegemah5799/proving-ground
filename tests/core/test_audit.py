from __future__ import annotations

import json

import pytest

from proving_ground.audit import GENESIS, AuditLog, verify
from proving_ground.canonical import canonical_bytes


def make_log(tmp_path, n=4):
    log = AuditLog(tmp_path / "decisions.jsonl")
    for i in range(n):
        log.append("hold", reason=f"reason {i}", candidate_hash=f"h{i}")
    return log


def test_chain_verifies_and_links(tmp_path):
    log = make_log(tmp_path)
    r = log.verify()
    assert r.ok and r.entries == 4 and r.head != GENESIS
    es = log.entries()
    assert es[0]["prev_entry_hash"] == GENESIS and es[1]["prev_entry_hash"] == es[0]["entry_hash"]


def test_edit_detected(tmp_path):
    log = make_log(tmp_path)
    lines = log.path.read_text().splitlines()
    e = json.loads(lines[1])
    e["reason"] = "tampered"
    lines[1] = json.dumps(e, sort_keys=True)
    log.path.write_text("\n".join(lines) + "\n")
    r = verify(log.path)
    assert not r.ok and r.broken_at_seq == 2 and "edited" in r.error


def test_deletion_detected(tmp_path):
    log = make_log(tmp_path)
    lines = log.path.read_text().splitlines()
    del lines[1]
    log.path.write_text("\n".join(lines) + "\n")
    r = verify(log.path)
    assert not r.ok and "gap" in r.error


def test_truncating_tail_is_visible_via_head_hash(tmp_path):
    log = make_log(tmp_path)
    head = log.verify().head
    lines = log.path.read_text().splitlines()[:-1]
    log.path.write_text("\n".join(lines) + "\n")
    assert verify(log.path).head != head  # tail truncation changes the head; anchor the head externally


def test_refuses_to_append_to_broken_chain(tmp_path):
    log = make_log(tmp_path)
    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    with pytest.raises(RuntimeError):
        log.append("hold", reason="x")


def test_floats_and_unknown_events_rejected(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    with pytest.raises(ValueError):
        log.append("explode", reason="x")
    with pytest.raises(TypeError):
        log.append("hold", reason="x", evidence=[{"v": 0.5}])


def test_canonical_serialization_is_documented_form():
    assert canonical_bytes({"b": 1, "a": [True, None, "é"]}) == '{"a":[true,null,"é"],"b":1}'.encode()


def test_unicode_line_separators_inside_entries_do_not_break_the_chain(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.append("hold", reason="a b\u0085c d")
    log.append("hold", reason="next")
    assert log.verify().ok and len(log.entries()) == 2
