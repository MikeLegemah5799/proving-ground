"""Environment sanity check used by `make check`."""

import importlib
import platform
import sys

ok = True
print(f"python {platform.python_version()} on {platform.system()}")
if sys.version_info < (3, 11):  # noqa: UP036
    print("  FAIL: Python 3.11+ required")
    ok = False
for mod in ("numpy", "pandas", "sklearn", "pandera", "fastapi", "pydantic", "lightgbm", "yaml"):
    try:
        importlib.import_module(mod)
        print(f"  ok   {mod}")
    except Exception as exc:  # noqa: BLE001
        hint = " (macOS: brew install libomp)" if mod == "lightgbm" and platform.system() == "Darwin" else ""
        print(f"  FAIL {mod}: {str(exc).splitlines()[0]}{hint}")
        ok = False
raise SystemExit(0 if ok else 1)
