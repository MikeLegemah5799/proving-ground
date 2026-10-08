"""Pydantic models for every JSON report the viewer reads.

Rules: the major version in ``schema_version`` must be supported (currently 1); unknown *minor* additions and
unknown fields are ignored. Unsupported majors fail the build with a clear message.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SUPPORTED_MAJOR = 1


class SchemaVersionError(ValueError):
    pass


def check_version(obj: dict[str, Any], kind: str, source: str = "") -> None:
    v = str(obj.get("schema_version", ""))
    try:
        major = int(v.split(".")[0])
    except ValueError:
        raise SchemaVersionError(f"{kind} {source}: missing or malformed schema_version {v!r}") from None
    if major != SUPPORTED_MAJOR:
        raise SchemaVersionError(
            f"{kind} {source}: unsupported schema_version {v!r}; this viewer understands major version {SUPPORTED_MAJOR}.x. "
            "Rebuild the report with a matching proving-ground version or upgrade the viewer.")


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class AuditEntry(_M):
    seq: int
    ts: str
    event: str
    candidate_hash: str | None = None
    champion_hash_before: str | None = None
    champion_hash_after: str | None = None
    git_sha: str | None = None
    data_version: str | None = None
    gate_config_hash: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""
    actor: str = ""
    synthetic: bool = False
    prev_entry_hash: str
    entry_hash: str


class CheckM(_M):
    name: str
    status: str
    value: Any = None
    threshold: Any = None
    ci: list[float] | None = None
    detail: str = ""
    champion_value: float | None = None


class GateM(_M):
    id: str
    name: str
    status: str
    checks: list[CheckM] = Field(default_factory=list)
    note: str = ""


class GateReport(_M):
    schema_version: str
    candidate_hash: str
    project: str = ""
    family: str = ""
    champion_hash: str | None = None
    created: str = ""
    overall: str
    failed_gates: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    gate_config_hash: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_ci: dict[str, list[float]] = Field(default_factory=dict)
    champion_metrics: dict[str, float] | None = None
    calibration: dict[str, float] = Field(default_factory=dict)
    calibration_detail: dict[str, Any] | None = None
    slices: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    gates: list[GateM] = Field(default_factory=list)
    synthetic: bool = False


class ShadowSummary(_M):
    schema_version: str
    candidate_hash: str
    champion_hash: str | None = None
    verdict: str
    requests: int = 0
    min_requests: int = 0
    agreement_rate: float | None = None
    spearman: float | None = None
    mean_latency_delta_ms: float | None = None
    exception_rate: float | None = None
    error_vs_champion: dict[str, Any] | None = None
    compare_metric: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)
    synthetic: bool = True


class CanarySummary(_M):
    schema_version: str
    candidate_hash: str
    champion_hash: str | None = None
    verdict: str
    canary_pct: int = 0
    requests_total: int = 0
    requests_canary: int = 0
    requests_control: int = 0
    error_rate: float | None = None
    p95_latency_ms: float | None = None
    prediction_psi_vs_shadow: float | None = None
    error_vs_control: dict[str, Any] | None = None
    compare_metric: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)
    aborted: bool = False
    synthetic: bool = True


class DriftWindow(_M):
    schema_version: str
    window: int
    level: str
    rows: int = 0
    champion_hash: str | None = None
    synthetic: bool = False
    scenarios: list[str] = Field(default_factory=list)
    features: dict[str, dict[str, Any]] = Field(default_factory=dict)
    prediction: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    label_drift: dict[str, Any] | None = None
    insufficient_data: bool = False
    window_start: str | None = None
    window_end: str | None = None
    html: str | None = None
    rollback_recommended: bool = False


class DriftTimeline(_M):
    schema_version: str
    run: str
    synthetic: bool = False
    windows: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioMatrix(_M):
    schema_version: str
    synthetic: bool = True
    scenarios: list[dict[str, Any]] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)


class Incident(_M):
    schema_version: str
    scenario: str
    synthetic: bool = True
    note: str = ""
    phases: list[dict[str, Any]] = Field(default_factory=list)
    audit_events: list[dict[str, Any]] = Field(default_factory=list)
    chain_ok: bool = True


class ModelComparison(_M):
    schema_version: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    note: str = ""
