from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..candidate import Candidate
from ..evaluation import HoldoutEval
from ..interface import DataBundle, ModelProject

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"


@dataclass
class Check:
    name: str
    status: str
    value: float | str | None = None
    threshold: float | str | None = None
    ci: list[float] | None = None
    detail: str = ""
    champion_value: float | None = None


@dataclass
class GateResult:
    id: str
    name: str
    status: str = PASS
    checks: list[Check] = field(default_factory=list)
    note: str = ""

    def add(self, c: Check) -> None:
        self.checks.append(c)
        if c.status == FAIL:
            self.status = FAIL
        elif c.status == WARN and self.status == PASS:
            self.status = WARN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateContext:
    project: ModelProject
    bundle: DataBundle
    candidate: Candidate
    eval: HoldoutEval
    cfg: dict[str, Any]
    report_dir: Path
    champion: Candidate | None = None
    champion_eval: HoldoutEval | None = None
    seed: int = 0
    n_boot: int = 200
    retrain: Any = None  # callable () -> Candidate, for G7
    card_text: str | None = None
