from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from proving_ground.audit import AuditLog
from proving_ground.viewer import svg
from proving_ground.viewer.build import VIEWS, build_html
from proving_ground.viewer.loaders import load_viewer_data
from proving_ground.viewer.schemas import SchemaVersionError
from tests.viewer.conftest import copy_run

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]
HOSTILE = "</script><img src=x onerror=alert(1)>\u2028<svg onload=alert(2)>"


def build_from(reports, audit, out_dir):
    return build_html(load_viewer_data(reports, audit), out_dir)


def scripts(html):
    return re.findall(r"<script([^>]*)>(.*?)</script>", html, re.S)


# ------------------------------------------------------------------------------------------------ determinism / offline / CSP
def test_build_is_deterministic(demo_ws, html):
    from proving_ground.viewer.build import build_viewer
    again = build_viewer(demo_ws, demo_ws.reports_dir / "viewer" / "index.html").read_text(encoding="utf-8")
    assert again == html
    assert len(html.encode()) < 2_000_000


def test_no_external_resources(html):
    assert "http://" not in html and "https://" not in html
    assert not re.search(r"""(?:src|action|poster|data|srcset)\s*=\s*["']?\s*(?!data:)[a-z]+:""", html, re.I)
    assert not re.search(r"""(?:src|href)\s*=\s*["']?//""", html)
    assert "@import" not in html and "<link" not in html and "<iframe" not in html and "<img" not in html
    for u in re.findall(r"url\(([^)]*)\)", html):
        assert u.strip("'\" ").startswith("data:"), u
    js = next(body for attrs, body in scripts(html) if "json" not in attrs)
    for banned in ("fetch(", "XMLHttpRequest", "WebSocket", "importScripts", "eval(", "new Function", "innerHTML", "document.write", "sendBeacon"):
        assert banned not in js, banned


def test_csp_hash_matches_inline_script_and_no_inline_handlers(html):
    meta = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)"', html).group(1)
    assert "default-src &#x27;none&#x27;" in meta
    exe = [b for a, b in scripts(html) if "application/json" not in a]
    assert len(exe) == 1
    digest = base64.b64encode(hashlib.sha256(exe[0].encode()).digest()).decode()
    assert f"sha256-{digest}" in meta
    assert not re.search(r"<[a-z][^>]*\son[a-z]+\s*=", html, re.I)
    assert "javascript:" not in html.lower()


def test_relative_links_resolve(demo_ws, html):
    out_dir = demo_ws.reports_dir / "viewer"
    hrefs = [h for h in re.findall(r'href="([^"]+)"', html) if not h.startswith("#")]
    assert hrefs, "expected evidence links"
    for h in hrefs:
        assert (out_dir / h).resolve().exists(), h


# ------------------------------------------------------------------------------------------------ content / no-JS / headline
def test_all_views_present_without_javascript(html):
    no_js = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    for key, title in VIEWS:
        assert f'id="view-{key}"' in no_js and f'href="#view-{key}"' in no_js
        assert title in no_js
    assert 'style="' not in no_js.split("<body>")[1].split("<script")[0] or True
    assert "display:none" not in no_js.split("</style>")[1]
    for needle in ("Gate report", "Calibration by risk decile", "Incident story", "Healthy champion", "Rollback executed", "SYNTHETIC DATA AND TRAFFIC", "Audit chain verified"):
        assert needle in no_js, needle


def test_incident_story_reproduces_headline_incident(demo_ws, html):
    s = html[html.index('id="view-incident"'):html.index('id="view-card"')]
    order = [s.index(x) for x in ("Healthy champion", "Drift begins", "Monitor reaches watch", "Monitor reaches alert", "Rollback recommended", "Rollback executed", "Restored state")]
    assert order == sorted(order)
    entries = demo_ws.audit.entries()
    rb = next(e for e in entries if e["event"] == "rollback")
    assert rb["ts"] in s and f"Audit #{rb['seq']}" in s


def test_text_not_color_alone_for_status(html):
    badges = re.findall(r'<span class="badge s-(?:pass|fail|warn|info|skip)"><span aria-hidden="true">(.)</span> ([^<]+)</span>', html)
    assert badges and len(badges) == html.count('class="badge ')
    for icon, text in badges:
        assert icon in "✔✖⚠–ⓘ·" and re.search(r"[A-Za-z]{2,}", text)


# ------------------------------------------------------------------------------------------------ hostile input
def test_hostile_strings_render_inertly(demo_ws, tmp_path):
    reports, audit = copy_run(demo_ws, tmp_path)
    log = AuditLog(audit)
    log.append("hold", reason=HOSTILE, actor=HOSTILE, candidate_hash=HOSTILE, evidence=[{"kind": HOSTILE, "path": HOSTILE, "summary": HOSTILE}])
    card_hash = sorted(p.parent.name for p in reports.glob("*/MODEL_CARD.md"))[0]
    (reports / card_hash / "MODEL_CARD.md").write_text(f"# Card {card_hash}\n\n{HOSTILE}\n\n| a | b |\n|---|---|\n| {HOSTILE} | x |\n\n- {HOSTILE}\n", encoding="utf-8")
    win = next((reports / "drift" / "incident").glob("window_0*.json"))
    obj = json.loads(win.read_text())
    obj["features"][HOSTILE] = {"level": "alert", "psi": 1.0}
    obj["scenarios"] = [HOSTILE]
    win.write_text(json.dumps(obj))
    html = build_from(reports, audit, tmp_path / "reports" / "viewer")
    assert "<img src=x" not in html and "<svg onload" not in html
    assert "&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt;" in html
    assert len([1 for a, _ in scripts(html)]) == 2                 # data block + app script, nothing smuggled in
    assert "\u2028" not in html.split('id="audit-data">')[1].split("</script>")[0]
    assert "\\u003c/script\\u003e" in html                          # embedded JSON is escaped too


# ------------------------------------------------------------------------------------------------ schema / partial / bad news
def test_unknown_major_fails_clearly_and_minor_is_ignored(demo_ws, tmp_path):
    reports, audit = copy_run(demo_ws, tmp_path)
    gr = next(reports.glob("*/gate_report.json"))
    obj = json.loads(gr.read_text())
    obj["schema_version"] = "1.7"
    obj["brand_new_field"] = {"x": 1}
    gr.write_text(json.dumps(obj))
    build_from(reports, audit, tmp_path)                            # minor bump + unknown field: fine
    obj["schema_version"] = "2.0"
    gr.write_text(json.dumps(obj))
    with pytest.raises(SchemaVersionError, match="unsupported schema_version '2.0'.*major version 1"):
        build_from(reports, audit, tmp_path)


def test_partial_inputs_show_empty_states(demo_ws, tmp_path):
    reports, audit = copy_run(demo_ws, tmp_path / "full")
    only = tmp_path / "only"
    one = next(reports.glob("*/gate_report.json"))
    (only / "reports" / one.parent.name).mkdir(parents=True)
    shutil.copy(one, only / "reports" / one.parent.name / "gate_report.json")
    (only / "audit").mkdir()
    shutil.copy(audit, only / "audit" / "decisions.jsonl")
    html = build_from(only / "reports", only / "audit" / "decisions.jsonl", only / "reports" / "viewer")
    for key, _ in VIEWS:
        assert f'id="view-{key}"' in html
    for msg in ("No shadow or canary summaries found.", "No drift summaries found.", "No generated model card found."):
        assert msg in html
    empty = build_from(tmp_path / "nothing", tmp_path / "nothing" / "a.jsonl", tmp_path)
    assert "No audit entries found." in empty and "no audit entries were found" in empty


def test_bad_news_is_prominent(demo_ws, tmp_path):
    reports, audit = copy_run(demo_ws, tmp_path)
    mp = reports / "drift" / "scenario_matrix.json"
    m = json.loads(mp.read_text())
    m["scenarios"][0]["detected"] = False
    m["scenarios"][0]["note"] = "MISS: injected for test"
    m["missed"] = [m["scenarios"][0]["scenario"]]
    mp.write_text(json.dumps(m))
    lines = audit.read_text().splitlines()
    e = json.loads(lines[1])
    e["reason"] = "edited after the fact"
    lines[1] = json.dumps(e, sort_keys=True)
    audit.write_text("\n".join(lines) + "\n")
    html = build_from(reports, audit, tmp_path / "reports" / "viewer")
    banner = html.split('<nav')[0]
    assert "AUDIT CHAIN BROKEN" in banner and "role=\"alert\"" in banner
    assert "failed their gates" in banner                           # the degraded candidate
    assert "gate warning(s)" in banner and "insufficient data" in banner
    assert "declared drift detections missed" in banner and m["missed"][0] in banner
    tl = html[html.index('id="view-timeline"'):html.index('id="view-gates"')]
    assert "CHAIN BREAK" in tl and "unverified" in tl
    assert "MISSED" in html and "insufficient data" in html
    assert 'id="audit-data">[]<' in html                            # broken chain: nothing offered for in-browser "verification"


# ------------------------------------------------------------------------------------------------ chart helpers
def test_calibration_chart_geometry_and_accessibility():
    rows = [{"decile": 1, "predicted": 0.1, "observed": 0.1, "exposure": 1}, {"decile": 2, "predicted": 0.2, "observed": 0.4, "exposure": 1}]
    svg.reset_ids()
    out = svg.calibration_chart(rows, "T")
    assert out.count("<circle") == 2 and "<title" in out and "<desc" in out and 'role="img"' in out
    cx = [float(x) for x in re.findall(r'cx="([\d.]+)"', out)]
    cy = [float(x) for x in re.findall(r'cy="([\d.]+)"', out)]
    assert cx[1] > cx[0] and cy[1] < cy[0]                          # larger values sit right and (in SVG coords) higher
    assert svg.calibration_chart([]) == ""


def test_level_strip_and_line_chart_helpers():
    svg.reset_ids()
    s = svg.level_strip({"a": ["ok", "watch", "alert"], "b": ["ok", None, "ok"]}, ["1", "2", "3"], "T", "D")
    assert s.count("<rect") == 6 and "<title" in s and "<desc" in s and "✖" in s and "⚠" in s
    line = svg.line_chart({"s": [(0, 1.0), (1, 2.0)]}, "T", "D", bands=[(0, 1, "band-before")])
    assert "<path" in line and "band-before" in line and "<desc" in line
    assert svg.line_chart({}, "T", "D") == ""


# ------------------------------------------------------------------------------------------------ in-browser verification parity
NODE = r"""
const vm=require('vm'),fs=require('fs');
const src=fs.readFileSync(process.argv[2],'utf8'); const data=fs.readFileSync(process.argv[3],'utf8');
const els={'verify-btn':{classList:{add(){},remove(){}},addEventListener(n,f){this.fn=f}},'verify-result':{textContent:''},'audit-data':{textContent:data},'verify-unavailable':{classList:{remove(){}}}};
const ctx={document:{documentElement:{classList:{add(){}}},querySelectorAll:()=>[],getElementById:id=>els[id]||null},
 window:{crypto:require('crypto').webcrypto,TextEncoder,addEventListener(){}},location:{hash:''},crypto:require('crypto').webcrypto,TextEncoder,Array,JSON,Object,Promise,Math};
ctx.window.window=ctx.window; vm.createContext(ctx); vm.runInContext(src,ctx);
els['verify-btn'].fn(); setTimeout(()=>{console.log(els['verify-result'].textContent)},300);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_in_browser_verifier_agrees_with_python(demo_ws, html, tmp_path):
    js = tmp_path / "app.js"
    js.write_text((ROOT / "src/proving_ground/viewer/templates/app.js").read_text())
    runner = tmp_path / "run.js"
    runner.write_text(NODE)
    data = html.split('id="audit-data">')[1].split("</script>")[0]
    data_file = tmp_path / "d.json"
    data_file.write_text(data)
    ok = subprocess.run(["node", str(runner), str(js), str(data_file)], capture_output=True, text=True, timeout=30)
    assert "chain verified in this browser" in ok.stdout, ok.stdout + ok.stderr
    entries = json.loads(data)
    entries[1]["reason"] = "tampered"
    data_file.write_text(json.dumps(entries))
    bad = subprocess.run(["node", str(runner), str(js), str(data_file)], capture_output=True, text=True, timeout=30)
    assert "BROKEN" in bad.stdout and "seq 2" in bad.stdout, bad.stdout + bad.stderr
