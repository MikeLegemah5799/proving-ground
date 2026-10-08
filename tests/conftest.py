from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from proving_ground.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def toy_ws(tmp_path):
    cdir = tmp_path / "config"
    cdir.mkdir()
    shutil.copy(ROOT / "examples/toy/gates.yaml", cdir / "gates.yaml")
    shutil.copy(ROOT / "examples/toy/release.yaml", cdir / "release.yaml")
    shutil.copy(ROOT / "config/monitoring.yaml", cdir / "monitoring.yaml")
    (cdir / "project.yaml").write_text("project: examples.toy.project:ToyProject\noptions: {}\ntraining: {seed: 5}\n")
    from examples.toy.project import ToyProject
    return Workspace.from_config(tmp_path, project=ToyProject({}))


def make_toy_ws(tmp_path, **options):
    from examples.toy.project import ToyProject
    cdir = tmp_path / "config"
    cdir.mkdir(exist_ok=True)
    for n in ("gates", "release"):
        shutil.copy(ROOT / f"examples/toy/{n}.yaml", cdir / f"{n}.yaml")
    shutil.copy(ROOT / "config/monitoring.yaml", cdir / "monitoring.yaml")
    (cdir / "project.yaml").write_text("project: examples.toy.project:ToyProject\ntraining: {seed: 5}\n")
    return Workspace.from_config(tmp_path, project=ToyProject(options))
