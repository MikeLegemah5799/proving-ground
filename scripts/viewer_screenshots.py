"""Capture README/blog screenshots of the report viewer. Needs the optional `dev` extra (Playwright)."""

from __future__ import annotations

import sys
from pathlib import Path

VIEWS = {"overview": "overview", "gates": "gate-report", "drift": "drift", "incident": "incident-story"}


def main(viewer: str = "reports/viewer/index.html", out: str = "docs/img") -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed: pip install -e '.[dev]' && playwright install chromium", file=sys.stderr)
        return 1
    url = Path(viewer).resolve().as_uri()
    Path(out).mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1200, "height": 900})
        for key, name in VIEWS.items():
            page.goto(f"{url}#view-{key}")
            page.wait_for_timeout(150)
            page.screenshot(path=f"{out}/{name}.png", full_page=True)
            print("wrote", f"{out}/{name}.png")
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
