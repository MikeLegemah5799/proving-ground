from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np

from proving_ground.serving.app import bucket
from proving_ground.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]


def test_core_never_imports_examples():
    """Invariant 12: nothing in template core imports from examples/."""
    bad = []
    for p in (ROOT / "src" / "proving_ground").rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            bad += [f"{p.name}: {n}" for n in names if n == "examples" or n.startswith("examples.")]
    # demo.py may *optionally* import the claims comparison helper; that is the only allowed lazy import
    bad = [b for b in bad if not b.startswith("demo.py: examples.claims_frequency.compare")]
    assert not bad, bad


def test_demo_is_the_only_example_aware_module():
    hits = [p.name for p in (ROOT / "src" / "proving_ground").rglob("*.py") if "examples." in p.read_text() and "import" in p.read_text()]
    assert set(hits) <= {"demo.py", "workspace.py", "cli.py"} | {p.name for p in (ROOT / "src" / "proving_ground").rglob("*.py") if "examples.claims" in p.read_text()}


def test_stable_hash_routing_is_consistent_and_roughly_uniform():
    keys = [f"policy-{i}" for i in range(20000)]
    assert [bucket(k) for k in keys[:50]] == [bucket(k) for k in keys[:50]]
    share = np.mean([bucket(k) < 10 for k in keys])
    assert 0.085 < share < 0.115


def test_make_targets_referenced_in_docs_exist():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "docs_check.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_config_parsing(toy_ws: Workspace):
    assert toy_ws.gates_cfg()["g2_floors"]["r2"]["min"] == 0.8
    assert len(toy_ws.gate_config_hash()) == 16
    h1 = toy_ws.gate_config_hash()
    toy_ws.gates_path.write_text(toy_ws.gates_path.read_text() + "\n# a comment changes nothing\n")
    assert toy_ws.gate_config_hash() == h1
