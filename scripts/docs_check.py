"""Fail if README/docs mention a `make <target>` that the Makefile does not define."""

import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
targets = set(re.findall(r"^([a-z][a-z-]*):", (root / "Makefile").read_text(), re.M))
bad = []
files = [root / "README.md", root / "TEMPLATE_GUIDE.md", *sorted((root / "docs").glob("*.md")), *sorted((root / "examples").rglob("*.md"))]
for f in files:
    if not f.exists():
        continue
    for m in re.finditer(r"`make ([a-z][a-z-]*)", f.read_text()):
        if m.group(1) not in targets:
            bad.append(f"{f.relative_to(root)}: make {m.group(1)}")
if bad:
    print("unknown make targets referenced:\n  " + "\n  ".join(bad))
    sys.exit(1)
print(f"docs-check ok ({len(targets)} targets)")
