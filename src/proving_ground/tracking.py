"""Opt-in experiment tracking. The file registry stays the source of truth; MLflow (local SQLite) is a mirror.

Enable with ``tracking: mlflow`` in config/project.yaml or ``PG_TRACKING=mlflow``. Needs ``pip install .[tracking]``.
A tracking failure never blocks a release decision: it is reported and ignored.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .workspace import Workspace


def enabled(ws: Workspace) -> bool:
    return os.environ.get("PG_TRACKING", "").lower() == "mlflow" or getattr(ws, "tracking", "") == "mlflow"


def _mlflow(ws: Workspace):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{ws.root / 'mlflow.db'}")
    mlflow.set_experiment(ws.project.name)
    return mlflow


def log_gate_run(ws: Workspace, report: dict[str, Any]) -> None:
    if not enabled(ws):
        return
    try:
        mlflow = _mlflow(ws)
        with mlflow.start_run(run_name=report["candidate_hash"]):
            mlflow.log_params({"family": report["family"], "seed": report["provenance"]["seed"], "data_version": report["provenance"]["data_version"],
                               "gate_config_hash": report["gate_config_hash"]})
            mlflow.log_metrics({k: v for k, v in report["metrics"].items()})
            mlflow.set_tags({"overall": report["overall"], "git_sha": report["provenance"]["git_sha"], "synthetic": str(report["synthetic"])})
            mlflow.log_artifact(str(ws.report_dir(report["candidate_hash"]) / "gate_report.json"))
    except Exception as exc:  # noqa: BLE001
        print(f"[tracking] MLflow logging skipped: {exc}", file=sys.stderr)


def mirror_aliases(ws: Workspace) -> None:
    if not enabled(ws):
        return
    try:
        mlflow = _mlflow(ws)
        exp = mlflow.get_experiment_by_name(ws.project.name)
        client = mlflow.MlflowClient()
        for k, v in ws.registry.aliases().items():
            client.set_experiment_tag(exp.experiment_id, f"alias.{k}", v or "")
    except Exception as exc:  # noqa: BLE001
        print(f"[tracking] MLflow alias mirror skipped: {exc}", file=sys.stderr)
