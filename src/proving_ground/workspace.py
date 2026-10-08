"""Workspace: resolves config + paths and ties registry, audit log and project together."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit import AuditLog
from .interface import DataBundle, ModelProject
from .provenance import config_hash, git_sha, load_yaml, lockfile_hash
from .registry import Registry


def load_project_class(spec: str):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)


@dataclass
class Workspace:
    root: Path
    project: ModelProject
    gates_path: Path
    monitoring_path: Path
    release_path: Path
    registry: Registry
    audit: AuditLog
    reports_dir: Path
    logs_dir: Path
    seed: int = 0
    synthetic_traffic: bool = True
    tracking: str = ""
    reload_fn: Callable[[], None] | None = None
    health_fn: Callable[[], dict[str, Any]] | None = None
    _bundle: DataBundle | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, root: str | Path = ".", config_dir: str | Path | None = None, *, project: ModelProject | None = None,
                    overrides: dict[str, Any] | None = None) -> Workspace:
        root = Path(root).resolve()
        cdir = Path(config_dir) if config_dir else root / "config"
        if not cdir.is_absolute():
            cdir = root / cdir
        pcfg = load_yaml(cdir / "project.yaml")
        pcfg.update(overrides or {})
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if project is None:
            project = load_project_class(pcfg["project"])(pcfg.get("options") or {})
        paths = pcfg.get("paths", {})
        p = lambda k, d: (root / paths.get(k, d))  # noqa: E731
        return cls(
            root=root, project=project, gates_path=cdir / "gates.yaml", monitoring_path=cdir / "monitoring.yaml",
            release_path=cdir / "release.yaml", registry=Registry(p("registry", "registry")),
            audit=AuditLog(p("audit", "audit/decisions.jsonl")), reports_dir=p("reports", "reports"),
            logs_dir=p("logs", "logs"), seed=int(pcfg.get("training", {}).get("seed", 0)), tracking=str(pcfg.get("tracking", "")),
        )

    # --- config -----------------------------------------------------------------
    def gates_cfg(self) -> dict[str, Any]:
        return load_yaml(self.gates_path)

    def release_cfg(self) -> dict[str, Any]:
        return load_yaml(self.release_path)

    def monitoring_cfg(self) -> dict[str, Any]:
        return load_yaml(self.monitoring_path)

    def gate_config_hash(self) -> str:
        return config_hash(self.gates_path)

    def monitoring_config_hash(self) -> str:
        return config_hash(self.monitoring_path)

    # --- data -------------------------------------------------------------------
    def data_version(self) -> str:
        return str((getattr(self.project, "data_version", None) or (lambda: "v0"))())

    def bundle(self) -> DataBundle:
        if self._bundle is None:
            self._bundle = self.project.load_data(self.data_version())
        return self._bundle

    def report_dir(self, h: str) -> Path:
        return self.reports_dir / h

    def prov(self) -> dict[str, Any]:
        return {"git_sha": git_sha(self.root), "data_version": self.bundle().data_version, "seed": self.seed,
                "lockfile_hash": lockfile_hash(self.root), "gate_config_hash": self.gate_config_hash()}

    # --- serving hooks ------------------------------------------------------------
    def reload(self) -> None:
        from .tracking import mirror_aliases

        mirror_aliases(self)
        if self.reload_fn:
            self.reload_fn()
