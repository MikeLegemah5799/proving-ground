"""Build the single-file, offline, read-only report viewer.

Security model (Invariants 14-16): all data is HTML-escaped at build time, embedded JSON is serialized safely, the page
ships with a restrictive CSP whose script hash is computed here, there are no inline event handlers, no network
requests and no external resources. The viewer never writes to the registry, audit log or config.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any

from ..workspace import Workspace
from . import svg
from .loaders import ViewerData, load_viewer_data
from .schemas import GateReport

TEMPLATES = Path(__file__).parent / "templates"
VIEWS = [("overview", "Overview"), ("timeline", "Release timeline"), ("gates", "Gate report"), ("calibration", "Calibration and slices"),
         ("release", "Shadow and canary"), ("drift", "Drift"), ("incident", "Incident story"), ("card", "Model card"), ("about", "About and limitations")]

esc = lambda s: escape(str(s), quote=True)  # noqa: E731
short = lambda h: esc(h[:10]) if h else "none"  # noqa: E731


def safe_json(obj: Any) -> str:
    """JSON for embedding in a <script type=application/json> block: cannot terminate the block or inject markup."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


# ----------------------------------------------------------------------------------------------- small HTML helpers
STATUS_CLASS = {"pass": "pass", "ok": "pass", "warn": "warn", "watch": "warn", "fail": "fail", "alert": "fail", "skip": "skip",
                "insufficient": "info", "extend": "warn", "none": "skip"}
STATUS_ICON = {"pass": "✔", "ok": "✔", "warn": "⚠", "watch": "⚠", "fail": "✖", "alert": "✖", "skip": "–", "insufficient": "ⓘ", "extend": "⚠", "none": "·"}


def badge(status: str | None, text: str | None = None) -> str:
    st = (status or "none").lower()
    cls = STATUS_CLASS.get(st, "info")
    return f'<span class="badge s-{cls}"><span aria-hidden="true">{STATUS_ICON.get(st, "ⓘ")}</span> {esc(text or st)}</span>'


def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return esc(v)


def table(headers: list[str], rows: list[list[str]], *, sortable: bool = False, row_classes: list[str] | None = None, caption: str = "") -> str:
    cls = ' class="sortable"' if sortable else ""
    head = "".join(f"<th scope=\"col\">{esc(h)}</th>" for h in headers)
    body = []
    for i, r in enumerate(rows):
        rc = f' class="{row_classes[i]}"' if row_classes and row_classes[i] else ""
        body.append(f"<tr{rc}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    cap = f"<caption class=\"hidden\">{esc(caption)}</caption>" if caption else ""
    return f'<div class="tablewrap"><table{cls}>{cap}<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def empty(msg: str) -> str:
    return f'<p class="empty">{esc(msg)}</p>'


def kv(pairs: list[tuple[str, str]]) -> str:
    return "<dl class=\"kv\">" + "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


def md_to_html(md: str) -> str:
    """Tiny markdown renderer. Everything is escaped first; raw HTML in the source is shown as text, never rendered."""
    def inline(t: str) -> str:
        t = esc(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
        return t
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif m := re.match(r"^(#{1,4})\s+(.*)$", ln):
            lvl = min(len(m.group(1)) + 2, 5)
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
        elif ln.startswith("|"):
            blk = []
            while i < len(lines) and lines[i].startswith("|"):
                blk.append(lines[i])
                i += 1
            cells = [[c.strip() for c in b.strip().strip("|").split("|")] for b in blk if not re.match(r"^\|\s*:?-{2,}", b)]
            if cells:
                out.append('<div class="tablewrap"><table><thead><tr>' + "".join(f"<th>{inline(c)}</th>" for c in cells[0]) + "</tr></thead><tbody>"
                           + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in cells[1:]) + "</tbody></table></div>")
        elif ln.startswith(">"):
            blk = []
            while i < len(lines) and lines[i].startswith(">"):
                blk.append(lines[i].lstrip("> "))
                i += 1
            out.append(f"<blockquote>{inline(' '.join(blk))}</blockquote>")
        elif re.match(r"^\s*[-*]\s+", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|\||>|\s*[-*]\s)", lines[i]):
                para.append(lines[i])
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return '<div class="md">' + "".join(out) + "</div>"


class Ctx:
    def __init__(self, d: ViewerData, out_dir: Path):
        self.d, self.out_dir = d, out_dir

    def link(self, rel: str | None, text: str | None = None) -> str:
        """Relative link to an evidence file under the reports dir, only if it exists."""
        if not rel:
            return ""
        target = self.d.reports_dir / rel
        label = esc(text or rel)
        if not target.exists():
            return f"{label} <span class=\"mono\">(file not found)</span>"
        return f'<a href="{esc(os.path.relpath(target, self.out_dir).replace(os.sep, "/"))}">{label}</a>'


# ----------------------------------------------------------------------------------------------- state derivation
def derive_state(d: ViewerData) -> dict[str, Any]:
    champion = previous = None
    for e in d.audit:
        if e.event in ("promote", "rollback"):
            champion = e.champion_hash_after
            previous = e.champion_hash_before if e.event == "promote" else None
    rs = d.registry_state
    if rs.get("aliases"):
        champion = rs["aliases"].get("champion") or champion
        previous = rs["aliases"].get("previous_champion", previous)
    family = {g.candidate_hash: g.family for g in d.gate_reports}
    return {"champion": champion, "previous": previous, "candidate": (rs.get("aliases") or {}).get("candidate"),
            "stage": rs.get("stage"), "canary_pct": rs.get("canary_pct"), "family": family}


# ----------------------------------------------------------------------------------------------- banners
def banners(d: ViewerData) -> str:
    out = []
    synth = [n for n, v in (
        ("audit entries", any(e.synthetic for e in d.audit)), ("gate reports (synthetic dataset)", any(g.synthetic for g in d.gate_reports)),
        ("shadow/canary traffic", bool(d.shadow or d.canary)), ("drift windows", any(w.synthetic for _, ws in d.drift.values() for w in ws)),
        ("incident run", d.incident is not None)) if v]
    if synth:
        out.append('<div class="banner b-warn" role="note"><strong>⚠ SYNTHETIC DATA AND TRAFFIC.</strong> Simulated traffic replays held-out policies; '
                   "drift is injected on purpose. Nothing here describes real-world behavior. Synthetic content found in: "
                   + esc(", ".join(synth)) + ".</div>")
    else:
        out.append('<div class="banner b-info" role="note"><strong>ⓘ No synthetic flag found in these inputs.</strong> Check the About panel before drawing conclusions.</div>')
    c = d.chain
    if c is None or (c.entries == 0 and c.ok):
        out.append('<div class="banner b-info" role="status"><strong>ⓘ Audit chain:</strong> no audit entries were found, so there is nothing to verify.</div>')
    elif c.ok:
        out.append(f'<div class="banner b-ok" role="status"><strong>✔ Audit chain verified at build time:</strong> {c.entries} entries, head <code>{esc(c.head[:16])}</code>. '
                   "A local file is tamper-evident, not tamper-proof.</div>")
    else:
        out.append(f'<div class="banner b-fail" role="alert"><strong>✖ AUDIT CHAIN BROKEN:</strong> {esc(c.error)}. Do not trust the timeline past entry {esc(c.broken_at_seq)}.</div>')
    bad = [g for g in d.gate_reports if g.overall != "pass"]
    warns = sum(len(g.warnings) for g in d.gate_reports)
    items = []
    if bad:
        items.append(f"{len(bad)} candidate(s) failed their gates: " + ", ".join(f"<code>{short(g.candidate_hash)}</code> ({esc('+'.join(g.failed_gates))})" for g in bad))
    if warns:
        items.append(f"{warns} gate warning(s) (including slices flagged <em>insufficient data</em> or with wide confidence intervals)")
    ab = [e for e in d.audit if e.event == "canary_abort"]
    if ab:
        items.append(f"{len(ab)} canary abort(s)")
    missed = (d.matrix.missed if d.matrix else [])
    if missed:
        items.append("declared drift detections missed: " + esc(", ".join(missed)))
    if d.incident and not d.incident.chain_ok:
        items.append("the incident run recorded a broken audit chain")
    if items:
        out.append('<div class="banner b-fail" role="alert"><strong>✖ Bad news is shown, not hidden:</strong><ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>")
    for n in d.notes:
        out.append(f'<div class="banner b-warn"><strong>⚠ Input problem:</strong> {esc(n)}</div>')
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------- views
def view_overview(c: Ctx) -> str:
    d, st = c.d, derive_state(c.d)
    fam = lambda h: esc(st["family"].get(h, "unknown")) if h else "–"  # noqa: E731
    ch = (f'<code>{esc(st["champion"])}</code> ({fam(st["champion"])})') if st["champion"] else "none yet"
    pv = (f'<code>{esc(st["previous"])}</code> ({fam(st["previous"])})') if st["previous"] else "none"
    cd = (f'<code>{esc(st["candidate"])}</code> ({fam(st["candidate"])})') if st["candidate"] else "none"
    last = d.audit[-1] if d.audit else None
    ld = (f'{badge(last.event)} seq {last.seq}, {esc(last.ts)}: {esc(last.reason)}') if last else "no decisions recorded"
    counts = {}
    for e in d.audit:
        counts[e.event] = counts.get(e.event, 0) + 1
    ev = ", ".join(f"{esc(k)}: {v}" for k, v in sorted(counts.items())) or "–"
    rt = d.manifest.get("seconds")
    pairs = [("Champion", ch), ("Previous champion", pv), ("Candidate", cd),
             ("Stage", esc(st["stage"]) + (f" (canary {esc(st['canary_pct'])}%)" if st.get("canary_pct") else "") if st["stage"] else "unknown (no registry state supplied)"),
             ("Latest decision", ld), ("Audit events", ev),
             ("Audit chain", badge("pass", f"verified ({d.chain.entries} entries)") if d.chain and d.chain.ok and d.chain.entries else
              (badge("fail", "BROKEN") if d.chain and not d.chain.ok else "no entries")),
             ("Candidates gated", f"{len(d.gate_reports)} ({sum(1 for g in d.gate_reports if g.overall == 'pass')} passed, {sum(1 for g in d.gate_reports if g.overall != 'pass')} rejected)")]
    if rt:
        pairs.append(("Demo runtime", f"{rt / 60:.1f} min"))
    body = "<h2>Overview</h2>" + kv(pairs)
    if d.comparison:
        rows = [[esc(r["name"]), f"<code>{short(r['hash'])}</code>", fmt(r["poisson_deviance"], 5), f"[{fmt(r['dev_ci'][0], 4)}, {fmt(r['dev_ci'][1], 4)}]",
                 fmt(r["deviance_skill"], 3), fmt(r["gini"], 3), f"[{fmt(r['gini_ci'][0], 3)}, {fmt(r['gini_ci'][1], 3)}]"] for r in d.comparison.rows]
        body += "<h3>Model comparison (same holdout, 95% bootstrap CIs)</h3>" + table(
            ["model", "candidate", "deviance", "deviance CI", "skill vs constant", "Gini", "Gini CI"], rows) + f"<p>{esc(d.comparison.note)}</p>"
    return body


def view_timeline(c: Ctx) -> str:
    d = c.d
    if not d.audit:
        return "<h2>Release timeline</h2>" + empty("No audit entries found.")
    bad = d.chain.broken_at_seq if d.chain and not d.chain.ok else None
    ev = [{"kind": e.event, "seq": e.seq, "broken": bad is not None and e.seq >= bad} for e in d.audit]
    chart = svg.timeline_svg(ev, "Release timeline", "Audit events in sequence order: " + ", ".join(f"#{e.seq} {e.event}" for e in d.audit))
    rows, cls = [], []
    for e in d.audit:
        broken = bad is not None and e.seq >= bad
        evid = "".join(f"<li>{esc(x.get('kind', ''))}: {c.link(x.get('path'), x.get('path') or 'n/a')} {esc(x.get('summary', ''))}</li>" for x in e.evidence)
        rows.append([str(e.seq), esc(e.ts), badge(e.event), esc(e.actor), esc(e.reason) + (f'<details><summary>evidence ({len(e.evidence)})</summary><ul>{evid}</ul></details>' if e.evidence else ""),
                     f"<code>{short(e.candidate_hash)}</code>", f"<code>{short(e.champion_hash_before)}</code> → <code>{short(e.champion_hash_after)}</code>",
                     (badge("fail", "CHAIN BREAK" if e.seq == bad else "unverified") if broken else "") + ("synthetic" if e.synthetic else "")])
        cls.append("row-break" if broken else ("row-fail" if e.event in ("reject", "canary_abort") else ("row-warn" if e.event in ("hold", "rollback") else "")))
    return ("<h2>Release timeline</h2>" + chart + table(["#", "time (UTC)", "event", "actor", "reason and evidence", "candidate", "champion before → after", "flags"], rows,
                                                         row_classes=cls, caption="Audit events") + "<p>Hashes shortened to 10 characters; full values in the JSON evidence.</p>")


def _gate_table(g: GateReport) -> str:
    rows, cls = [], []
    for gate in g.gates:
        rows.append([f"<strong>{esc(gate.id)}</strong>", esc(gate.name), badge(gate.status), "", "", "", "", esc(gate.note)])
        cls.append("row-gate")
        for ck in gate.checks:
            insufficient = str(ck.value) == "insufficient data"
            st = "insufficient" if insufficient else ck.status
            ci = f"[{ck.ci[0]:.4g}, {ck.ci[1]:.4g}]" if ck.ci else "–"
            rows.append(["", esc(ck.name), badge(st), fmt(ck.value), esc(ck.threshold) if ck.threshold is not None else "–", ci, fmt(ck.champion_value), esc(ck.detail)])
            cls.append({"fail": "row-fail", "warn": "row-warn"}.get(ck.status, ""))
    return table(["gate", "check", "status", "measured", "threshold", "95% CI", "champion", "detail"], rows, row_classes=cls, caption=f"Gate checks for {g.candidate_hash}")


def view_gates(c: Ctx) -> str:
    d = c.d
    if not d.gate_reports:
        return "<h2>Gate report</h2>" + empty("No gate reports found.")
    out = ["<h2>Gate report</h2>"]
    order = sorted(d.gate_reports, key=lambda g: g.created, reverse=True)
    for i, g in enumerate(order):
        prov = g.provenance
        met = []
        for k, v in g.metrics.items():
            ci = g.metric_ci.get(k)
            champ = (g.champion_metrics or {}).get(k)
            met.append([esc(k), fmt(v, 5), f"[{ci[0]:.4g}, {ci[1]:.4g}]" if ci else "–", fmt(champ, 5) if g.champion_metrics else "no champion at the time"])
        out.append(f'<details {"open" if i < 3 else ""}><summary>Candidate <code>{esc(g.candidate_hash)}</code> ({esc(g.family)}) {badge(g.overall)}'
                   + (f" failed: {esc(', '.join(g.failed_gates))}" if g.failed_gates else "") + "</summary>")
        out.append(kv([("data version", f"<code>{esc(prov.get('data_version'))}</code>"), ("git", f"<code>{esc(prov.get('git_sha'))}</code>"), ("seed", esc(prov.get("seed"))),
                       ("gate config", f"<code>{esc(g.gate_config_hash)}</code>"), ("lockfile", f"<code>{esc(prov.get('lockfile_hash'))}</code>"),
                       ("evidence", c.link(f"{g.candidate_hash}/gate_report.json", "gate_report.json") + " · " + c.link(f"{g.candidate_hash}/gate_report.md", "gate_report.md"))]))
        out.append("<h4>Candidate vs champion (holdout, exposure-weighted)</h4>" + table(["metric", "candidate", "95% CI", "champion"], met))
        out.append("<h4>All checks</h4>" + _gate_table(g) + "</details>")
    return "".join(out)


def view_calibration(c: Ctx) -> str:
    d = c.d
    reps = [g for g in sorted(d.gate_reports, key=lambda g: g.created, reverse=True)]
    if not reps:
        return "<h2>Calibration and slices</h2>" + empty("No gate reports found, so there is no calibration or slice data.")
    out = ["<h2>Calibration and slices</h2>"]
    for i, g in enumerate(reps):
        det = g.calibration_detail or {}
        rows = det.get("deciles") or []
        out.append(f'<details {"open" if i < 2 else ""}><summary>Candidate <code>{short(g.candidate_hash)}</code> ({esc(g.family)}) {badge(g.overall)}</summary>')
        out.append(kv([(k, fmt(v, 4)) for k, v in g.calibration.items()]))
        if rows:
            out.append(svg.calibration_chart(rows, f"Calibration by risk decile, candidate {g.candidate_hash[:10]}"))
            out.append("<details><summary>Data table for the chart</summary>" + table(["decile", "exposure", "predicted", "observed"], [[str(r["decile"]), fmt(r["exposure"], 5), fmt(r["predicted"], 4), fmt(r["observed"], 4)] for r in rows]) + "</details>")
        else:
            out.append(empty("No decile detail in this report."))
        srows, cls = [], []
        for sname, lv in g.slices.items():
            for r in lv:
                small = r["n"] < r.get("min_samples", 0)
                ae = r["scores"].get("ae_ratio") if r.get("scores") else None
                ci = r.get("ci", {}).get("ae_ratio")
                srows.append([esc(sname), esc(r["level"]), str(r["n"]), fmt(ae, 3), f"[{ci[0]:.3g}, {ci[1]:.3g}]" if ci else "–",
                              badge("insufficient", "insufficient data") if small else ""])
                cls.append("row-info" if small else "")
        out.append("<h4>Slices (actual / expected claims; 1.0 is calibrated)</h4>" + (table(["slice", "level", "n", "A/E", "95% CI", "flag"], srows, sortable=True, row_classes=cls) if srows else empty("No slices declared.")) + "</details>")
    return "".join(out)


def view_release(c: Ctx) -> str:
    d = c.d
    hashes = sorted(set(d.shadow) | set(d.canary))
    if not hashes:
        return "<h2>Shadow and canary</h2>" + empty("No shadow or canary summaries found.")
    out = ["<h2>Shadow and canary</h2>", '<p class="mono">All traffic below is simulated (held-out policies replayed in random order).</p>']
    for h in hashes:
        out.append(f"<h3>Candidate <code>{esc(h)}</code></h3>")
        s, ca = d.shadow.get(h), d.canary.get(h)
        if s:
            err = s.error_vs_champion
            out.append("<h4>Shadow " + badge(s.verdict) + "</h4>" + kv([
                ("requests", f"{s.requests} (minimum {s.min_requests})"), ("agreement within tolerance", fmt(s.agreement_rate, 3)), ("rank correlation (Spearman)", fmt(s.spearman, 3)),
                ("mean latency delta (candidate − champion)", f"{fmt(s.mean_latency_delta_ms, 3)} ms"), ("exception rate", fmt(s.exception_rate, 3)),
                (f"{esc(s.compare_metric)} vs champion (labelled rows)", f"candidate {fmt(err['candidate'], 4)} vs champion {fmt(err['reference'], 4)} (n={err['n']})" if err else "no labels yet"),
                ("evidence", c.link(f"{h}/shadow_summary.json", "shadow_summary.json"))]))
            out.append(table(["check", "status", "value", "threshold"], [[esc(k["name"]), badge(k["status"]), fmt(k["value"], 4), esc(k["threshold"])] for k in s.checks]) if s.checks else "")
        else:
            out.append(empty("No shadow summary for this candidate."))
        if ca:
            out.append("<h4>Canary " + badge("fail" if ca.aborted else ca.verdict, "ABORTED" if ca.aborted else ca.verdict) + "</h4>" + kv([
                ("traffic", f"{ca.canary_pct}% canary: {ca.requests_canary} canary / {ca.requests_control} control requests"), ("error rate", fmt(ca.error_rate, 3)),
                ("p95 latency", f"{fmt(ca.p95_latency_ms, 3)} ms"), ("prediction PSI vs shadow baseline", fmt(ca.prediction_psi_vs_shadow, 3)),
                (f"{esc(ca.compare_metric)} vs control", f"candidate {fmt(ca.error_vs_control['candidate'], 4)} vs champion {fmt(ca.error_vs_control['reference'], 4)}" if ca.error_vs_control else "no labels yet"),
                ("evidence", c.link(f"{h}/canary_summary.json", "canary_summary.json"))]))
            out.append(table(["guardrail", "status", "value", "threshold"], [[esc(k["name"]), badge(k["status"]), fmt(k["value"], 4), esc(k["threshold"])] for k in ca.checks]) if ca.checks else "")
        else:
            out.append(empty("No canary summary for this candidate."))
    return "".join(out)


def view_drift(c: Ctx) -> str:
    d = c.d
    if not d.drift and not d.matrix:
        return "<h2>Drift</h2>" + empty("No drift summaries found.")
    out = ["<h2>Drift</h2>", '<p class="mono">Drift below is injected on purpose into simulated traffic (synthetic: true).</p>']
    for run, (_tl, wins) in sorted(d.drift.items()):
        if not wins:
            continue
        cols = [str(w.window) for w in wins]
        feats = sorted({f for w in wins for f in w.features})
        strip = {f: [w.features.get(f, {}).get("level") for w in wins] for f in feats}
        strip["prediction"] = [w.prediction.get("level") for w in wins]
        strip["validation failures"] = [w.validation.get("level") for w in wins]
        strip["labels (A/E)"] = [(w.label_drift or {}).get("level") for w in wins]
        strip["WINDOW"] = [w.level for w in wins]
        out.append(f"<h3>Run <code>{esc(run)}</code></h3>")
        out.append(svg.level_strip(strip, cols, f"Drift status per signal and window, run {run}", "Rows are signals, columns are time windows; each cell is ok, watch or alert.", cell=30))
        rows, cls = [], []
        for w in wins:
            ld = w.label_drift
            html = c.link(f"drift/{run}/{w.html}", "Evidently report") if w.html else "–"
            rows.append([str(w.window), badge(w.level), str(w.rows), esc(w.window_start or "–"), esc(", ".join(w.scenarios) or "none") + (" (synthetic)" if w.synthetic else ""),
                         fmt(w.prediction.get("psi"), 3), fmt(w.validation.get("failure_rate"), 3),
                         (f"{ld['relative_change']:.3f}{'' if ld.get('significant', True) else ' (n.s.)'}" if ld else "–"), c.link(f"drift/{run}/window_{w.window:02d}.json", "json") + " · " + html])
            cls.append({"alert": "row-fail", "watch": "row-warn"}.get(w.level, ""))
        out.append(table(["window", "level", "rows", "start (UTC)", "scenario", "prediction PSI", "validation failure rate", "label drift (rel. change)", "evidence"], rows, row_classes=cls, caption="Drift windows"))
        worst = max(wins, key=lambda w: ({"ok": 0, "watch": 1, "alert": 2}[w.level], w.window))
        frows = [[esc(f), badge(v["level"]), fmt(v.get("psi"), 3), fmt(v.get("ks_stat"), 3), fmt(v.get("new_category_rate"), 3), fmt(v.get("out_of_range_rate"), 3)] for f, v in sorted(worst.features.items())]
        out.append(f"<h4>Per-feature detail at the worst window ({worst.window})</h4>" + table(["feature", "level", "PSI", "KS statistic", "new-category rate", "out-of-range rate"], frows, sortable=True))
    if d.matrix:
        rows, cls = [], []
        for r in d.matrix.scenarios:
            rows.append([esc(r["scenario"]), esc(r["description"]), esc(f"{r['expected'].get('level')} on {r['expected'].get('signal')}" + (", watch before alert" if r["expected"].get("watch_before_alert") else "")),
                         esc(" → ".join(r["window_levels"])), badge("pass", "detected") if r["detected"] else badge("fail", "MISSED") + " " + esc(r.get("note", ""))])
            cls.append("" if r["detected"] else "row-fail")
        out.append("<h3>Scenario matrix: declared vs observed (synthetic)</h3>" + table(["scenario", "what changes", "declared expectation", "window levels", "result"], rows, row_classes=cls)
                   + (f"<p><strong>Missed:</strong> {esc(', '.join(d.matrix.missed))}</p>" if d.matrix.missed else "<p>Every declared expectation was met in this run. See docs/MONITORING.md for sensitivity limits.</p>"))
    return "".join(out)


def view_incident(c: Ctx) -> str:
    d = c.d
    inc = d.incident
    if inc is None:
        holds = [e for e in d.audit if e.event in ("hold", "rollback")]
        if not holds:
            return "<h2>Incident story</h2>" + empty("No incident evidence found (needs reports/incident/incident.json or hold/rollback audit entries).")
        return "<h2>Incident story</h2>" + empty("No incident.json; showing only audit entries.") + table(["#", "time", "event", "reason"], [[str(e.seq), esc(e.ts), badge(e.event), esc(e.reason)] for e in holds])
    ph = inc.phases
    first = lambda pred: next((p for p in ph if pred(p)), None)  # noqa: E731
    before, during, after = [p for p in ph if p["phase"] == "before"], [p for p in ph if p["phase"] == "during"], [p for p in ph if p["phase"] == "after"]
    onset = first(lambda p: (p.get("intensity") or 0) > 0)
    watch, alert = first(lambda p: p["level"] == "watch"), first(lambda p: p["level"] == "alert")
    hold = next((e for e in inc.audit_events if e["event"] == "hold"), None)
    rb = next((e for e in reversed(inc.audit_events) if e["event"] == "rollback"), None)

    def cal(p):
        k = (p or {}).get("calibration") or {}
        sc = k.get("scores", {})
        cb = k.get("calibration", {})
        return (f"A/E {fmt(sc.get('ae_ratio'), 3)}, decile calibration error {fmt(cb.get('decile_calibration_error'), 3)}, deviance {fmt(sc.get('poisson_deviance'), 4)} (n={k.get('n')})") if k else "no labelled rows"
    steps = []

    def step(cls, title, body):
        steps.append(f'<li class="st-{cls}"><strong>{title}</strong><br>{body}</li>')
    b0 = before[0] if before else None
    step("ok", "Healthy champion", f"Window {esc(b0['window'])}, champion <code>{short(b0['serving_champion'])}</code>, level {badge(b0['level'])}. {esc(cal(b0))}" if b0 else "No pre-drift window recorded.")
    step("warn", "Drift begins (synthetic)", f"Scenario <code>{esc(inc.scenario)}</code> starts at window {esc(onset['window'])} with intensity {fmt(onset.get('intensity'), 3)}; level {badge(onset['level'])}." if onset else "No drifted window recorded.")
    step("warn", "Monitor reaches watch", (f"Window {esc(watch['window'])} (intensity {fmt(watch.get('intensity'), 3)}): {badge('watch')}. {esc(cal(watch))}" if watch else
                                         "<strong>No watch level was observed:</strong> the monitor went straight from ok to alert in this run. That is a finding, not a pass."))
    step("fail", "Monitor reaches alert", f"Window {esc(alert['window'])} (intensity {fmt(alert.get('intensity'), 3)}, {esc(alert.get('start') or 'no timestamp')}): {badge('alert')}. {esc(cal(alert))}" if alert else "The monitor never reached alert.")
    step("fail", "Rollback recommended", f"Audit #{esc(hold['seq'])} at {esc(hold['ts'])}: {esc(hold['reason'])}" if hold else "No hold entry in the audit log.")
    step("warn", "Rollback executed", f"Audit #{esc(rb['seq'])} at {esc(rb['ts'])}: <code>{short(rb['champion_hash_before'])}</code> → <code>{short(rb['champion_hash_after'])}</code>. {esc(rb['reason'])}" if rb else "No rollback entry in the audit log.")
    a0 = after[0] if after else None
    step("ok", "Restored state", (f"Window {esc(a0['window'])}: serving <code>{short(a0['serving_champion'])}</code>, level {badge(a0['level'])}. {esc(cal(a0))}. "
                                  "The input drift is still present after rollback: rolling back restores the previous known-good model, it does not fix the data.") if a0 else "No post-rollback window recorded.")
    out = ["<h2>Incident story</h2>", '<p class="mono">Synthetic drift on simulated traffic, driven by the audit log and drift reports.</p>', "<ol class=\"steps\">" + "".join(steps) + "</ol>"]
    series = {"A/E (champion in service)": [(p["window"], p["calibration"]["scores"]["ae_ratio"]) for p in ph if p.get("calibration")]}
    if series["A/E (champion in service)"]:
        bands = []
        for name, ps in (("band-before", before), ("band-during", during), ("band-after", after)):
            if ps:
                bands.append((ps[0]["window"] - 0.5, ps[-1]["window"] + 0.5, name))
        out.append(svg.line_chart(series, "Actual / expected claims per window", "A/E for the serving champion by window; shaded bands mark before, during and after.", xlabel="window (bands: before | drift | after rollback)", ylabel="A/E", ref_line=1.0, bands=bands))
        out.append("<details><summary>Data table for the chart</summary>" + table(["window", "phase", "level", "serving champion", "A/E", "decile error", "rows"],
                   [[str(p["window"]), esc(p["phase"]), badge(p["level"]), f"<code>{short(p['serving_champion'])}</code>", fmt((p.get('calibration') or {}).get('scores', {}).get('ae_ratio'), 3),
                     fmt(((p.get('calibration') or {}).get('calibration') or {}).get('decile_calibration_error'), 3), str(p["rows"])] for p in ph]) + "</details>")
    for label, p in (("Before (healthy)", b0), ("At first alert", alert), ("After rollback", a0)):
        dec = (((p or {}).get("calibration") or {}).get("calibration") or {}).get("deciles")
        if dec:
            out.append(f"<h4>Calibration: {label}, window {esc(p['window'])}</h4>" + svg.calibration_chart(dec, f"Calibration by decile: {label}"))
    out.append(f"<p>{esc(inc.note)}</p>")
    return "".join(out)


def view_card(c: Ctx) -> str:
    d = c.d
    if not d.cards:
        return "<h2>Model card</h2>" + empty("No generated model card found.")
    st = derive_state(d)
    h = st["champion"] if st["champion"] in d.cards else (sorted(d.cards)[-1])
    others = [k for k in sorted(d.cards) if k != h]
    out = [f"<h2>Model card</h2><p>Card for candidate <code>{esc(h)}</code> (the model most recently in service among those with a card). Generated, not hand-written.</p>", md_to_html(d.cards[h])]
    for o in others:
        out.append(f"<details><summary>Card for candidate <code>{short(o)}</code></summary>{md_to_html(d.cards[o])}</details>")
    return "".join(out)


def view_about(c: Ctx) -> str:
    return ("<h2>About and limitations</h2><div class=\"card\"><ul>"
            "<li><strong>No timestamps in the dataset.</strong> The public frequency dataset has no date column, so a time-based split is impossible. Temporal generalization was not evaluated.</li>"
            "<li><strong>Traffic is simulated.</strong> \"Production\" traffic replays held-out policies in random order. Shadow and canary results demonstrate the machinery, not real production behavior.</li>"
            "<li><strong>Drift is synthetic.</strong> Every scenario is injected on purpose and tagged <code>synthetic: true</code>.</li>"
            "<li><strong>The audit log is a local file.</strong> It is tamper-<em>evident</em>, not tamper-<em>proof</em>: anyone who can rewrite the whole file can recompute the chain. Anchor the head hash somewhere you trust if that matters.</li>"
            "<li><strong>This page is read-only.</strong> It is generated from JSON reports and the audit log, never writes to them, makes no network requests and loads no external resources.</li>"
            "<li><strong>No compliance or fairness claim.</strong> Driver age and region can proxy for protected characteristics; no actuarial sign-off, pricing or underwriting advice is implied.</li>"
            "<li><strong>Verification in the browser.</strong> The audit chain is verified when this page is built. The button below recomputes it in your browser with Web Crypto; if your browser does not provide Web Crypto for <code>file://</code> pages, the button is hidden and the build-time result is the only one shown.</li>"
            "</ul></div><p><button type=\"button\" id=\"verify-btn\" class=\"noprint\">Verify audit chain in this browser</button><span id=\"verify-result\" role=\"status\"></span>"
            "<span id=\"verify-unavailable\" class=\"hidden\">Web Crypto is unavailable here, so in-browser verification is off. Only the build-time result (top of the page) applies.</span></p>")


BUILDERS = {"overview": view_overview, "timeline": view_timeline, "gates": view_gates, "calibration": view_calibration, "release": view_release,
            "drift": view_drift, "incident": view_incident, "card": view_card, "about": view_about}


def build_html(d: ViewerData, out_dir: Path) -> str:
    svg.reset_ids()
    c = Ctx(d, out_dir)
    js = (TEMPLATES / "app.js").read_text(encoding="utf-8")
    css = (TEMPLATES / "style.css").read_text(encoding="utf-8")
    csp_hash = base64.b64encode(hashlib.sha256(js.encode("utf-8")).digest()).decode()
    csp = (f"default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'sha256-{csp_hash}'; "
           "base-uri 'none'; form-action 'none'")
    nav = "".join(f'<li><a role="tab" id="tab-{k}" href="#view-{k}" aria-controls="view-{k}" aria-selected="{"true" if i == 0 else "false"}">{esc(t)}</a></li>' for i, (k, t) in enumerate(VIEWS))
    sections = "\n".join(
        f'<section class="view{" active" if i == 0 else ""}" id="view-{k}" role="tabpanel" aria-labelledby="tab-{k}" tabindex="0">{BUILDERS[k](c)}</section>' for i, (k, _) in enumerate(VIEWS))
    audit_json = safe_json([e.model_dump() for e in d.audit]) if d.chain and d.chain.ok else "[]"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{esc(csp)}">
<meta name="color-scheme" content="light dark">
<title>Proving Ground report</title>
<style>{css}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header><h1>Proving Ground <span class="sub">a gated model release template</span></h1><p class="sub">Release report. Static, offline, read-only.</p></header>
<div id="banners">{banners(d)}</div>
<nav aria-label="Report sections"><ul role="tablist">{nav}</ul></nav>
<main id="main">
{sections}
</main>
<footer>Generated from <code>audit/decisions.jsonl</code> and JSON reports by <code>proving-ground viewer build</code>. Built by Michael Legemah, mleg.tech. Synthetic demo data: see the About panel.</footer>
<script type="application/json" id="audit-data">{audit_json}</script>
<script>{js}</script>
</body>
</html>
"""


def build_viewer(ws: Workspace, out: Path) -> Path:
    out = Path(out)
    d = load_viewer_data(ws.reports_dir, ws.audit.path, ws.registry.root)
    html = build_html(d, out.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
