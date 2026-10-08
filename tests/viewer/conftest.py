from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from proving_ground.demo import run_demo
from proving_ground.viewer.build import build_viewer
from tests.claims.conftest import claims_ws

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def demo_ws(tmp_path_factory):
    """A real (fast, synthetic-data) end-to-end run: the viewer is tested against real artifacts, not hand-written ones."""
    tmp = tmp_path_factory.mktemp("demo")
    ws = claims_ws(tmp)
    shutil.copy(ROOT / "config" / "gates.synthetic.yaml", tmp / "config" / "gates.synthetic.yaml")
    assert run_demo(ws, fast=True) == 0
    return ws


@pytest.fixture(scope="session")
def html(demo_ws):
    return build_viewer(demo_ws, demo_ws.reports_dir / "viewer" / "index.html").read_text(encoding="utf-8")


def copy_run(demo_ws, dest: Path):
    """Independent copy of a demo run (reports + audit) that tests may mutate."""
    shutil.copytree(demo_ws.reports_dir, dest / "reports", ignore=shutil.ignore_patterns("viewer", "*.npz"))
    (dest / "audit").mkdir()
    shutil.copy(demo_ws.audit.path, dest / "audit" / "decisions.jsonl")
    return dest / "reports", dest / "audit" / "decisions.jsonl"
