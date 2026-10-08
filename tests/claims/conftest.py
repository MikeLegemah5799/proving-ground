from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from proving_ground.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]


def claims_ws(tmp: Path, **opts) -> Workspace:
    from examples.claims_frequency.project import ClaimsFrequencyProject
    c = tmp / "config"
    c.mkdir(exist_ok=True)
    for n in ("monitoring", "release", "project"):
        shutil.copy(ROOT / "config" / f"{n}.yaml", c / f"{n}.yaml")
    shutil.copy(ROOT / "config" / "gates.synthetic.yaml", c / "gates.yaml")
    options = {"data_source": "synthetic", "sample": 40000, **opts}
    return Workspace.from_config(tmp, project=ClaimsFrequencyProject(options))


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    return claims_ws(tmp_path_factory.mktemp("claims"))
