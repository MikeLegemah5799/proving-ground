"""Rollback: alias flip + hot reload + health verification. Idempotent."""

from __future__ import annotations

from typing import Any

from .workspace import Workspace


class RollbackError(Exception):
    pass


def rollback(ws: Workspace, reason: str, evidence: list[dict[str, Any]] | None = None, actor: str = "operator",
             synthetic: bool = False) -> dict[str, Any]:
    al, st = ws.registry.aliases(), ws.registry.state()
    champ, prev = al["champion"], al["previous_champion"]
    if prev is None:
        if st.get("rolled_back_from") and champ:
            return {"status": "already_rolled_back", "champion": champ, "rolled_back_from": st["rolled_back_from"]}
        raise RollbackError("no previous_champion to restore")
    if not ws.registry.exists(prev):
        raise RollbackError(f"previous champion {prev} is missing from the registry")
    ws.registry.set_aliases(champion=prev, previous_champion=None, candidate=None)
    ws.registry.set_state(stage="stable", canary_pct=0, rolled_back_from=champ)
    ws.reload()
    if ws.health_fn:
        served, via = ws.health_fn().get("champion_hash"), "GET /health"
    else:
        served, via = ws.registry.get_alias("champion"), "registry alias (no server attached)"
    ok = served == prev
    ev = (evidence or []) + [{"kind": "health_check", "path": "", "summary": f"served champion {served} via {via}; expected {prev}: {'OK' if ok else 'MISMATCH'}"}]
    entry = ws.audit.append("rollback", candidate_hash=champ, champion_hash_before=champ, champion_hash_after=prev,
                            git_sha=ws.prov()["git_sha"], data_version=ws.bundle().data_version,
                            gate_config_hash=ws.gate_config_hash(), evidence=ev, actor=actor, synthetic=synthetic,
                            reason=reason + ("" if ok else " [HEALTH VERIFICATION FAILED]"))
    if not ok:
        raise RollbackError(f"restored hash {prev} but server reports {served}")
    return {"status": "rolled_back", "champion": prev, "rolled_back_from": champ, "audit_seq": entry["seq"]}
