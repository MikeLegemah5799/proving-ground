"""Hand-rolled inline SVG helpers (no JS charting library). Output is deterministic and printable.

Colors are never inline: elements carry CSS classes (``s-ok``, ``s-watch``, ``s-alert``, ``series-a`` ...) that the
page stylesheet maps to theme variables, so charts follow light/dark mode. Every chart has <title> and <desc>.
"""

from __future__ import annotations

from html import escape as esc

ICON = {"ok": "✔", "pass": "✔", "watch": "⚠", "warn": "⚠", "alert": "✖", "fail": "✖", "skip": "–", "insufficient": "ⓘ", "none": "·"}
LETTER = {"ok": "ok", "watch": "watch", "alert": "ALERT"}


def f(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


_counter = [0]


def reset_ids() -> None:
    """Called at the start of every build so ids (and therefore the output bytes) are deterministic."""
    _counter[0] = 0


def _open(w: int, h: int, title: str, desc: str, cls: str = "chart") -> str:
    _counter[0] += 1
    n = _counter[0]
    return (f'<svg class="{cls}" role="img" viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet" '
            f'aria-labelledby="t{n} d{n}"><title id="t{n}">{esc(title)}</title><desc id="d{n}">{esc(desc)}</desc>')


def calibration_chart(rows: list[dict], title: str = "Calibration by risk decile", w: int = 420, h: int = 300) -> str:
    """Reliability chart: predicted vs observed frequency per decile, with the y = x line."""
    if not rows:
        return ""
    pad_l, pad_b, pad_t, pad_r = 44, 36, 16, 12
    vals = [r["predicted"] for r in rows] + [r["observed"] for r in rows]
    hi = max(vals) * 1.08 or 1.0
    px = lambda v: pad_l + v / hi * (w - pad_l - pad_r)        # noqa: E731
    py = lambda v: h - pad_b - v / hi * (h - pad_b - pad_t)    # noqa: E731
    desc = "; ".join(f"decile {r['decile']}: predicted {r['predicted']:.4f}, observed {r['observed']:.4f}" for r in rows)
    out = [_open(w, h, title, desc, "chart cal")]
    out.append(f'<line class="axis" x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}"/><line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}"/>')
    out.append(f'<line class="diag" x1="{f(px(0))}" y1="{f(py(0))}" x2="{f(px(hi))}" y2="{f(py(hi))}"/>')
    for t in range(0, 5):
        v = hi * t / 4
        out.append(f'<text class="tick" x="{pad_l - 6}" y="{f(py(v) + 3)}" text-anchor="end">{v:.3f}</text>'
                   f'<text class="tick" x="{f(px(v))}" y="{h - pad_b + 14}" text-anchor="middle">{v:.3f}</text>')
    for r in rows:
        out.append(f'<circle class="series-a" cx="{f(px(r["predicted"]))}" cy="{f(py(r["observed"]))}" r="4"><title>decile {r["decile"]}: predicted {r["predicted"]:.4f}, observed {r["observed"]:.4f}</title></circle>')
    out.append(f'<text class="label" x="{w // 2}" y="{h - 4}" text-anchor="middle">predicted frequency</text>'
               f'<text class="label" transform="rotate(-90 12 {h // 2})" x="12" y="{h // 2}" text-anchor="middle">observed frequency</text></svg>')
    return "".join(out)


def line_chart(series: dict[str, list[tuple[float, float]]], title: str, desc: str, *, xlabel: str = "", ylabel: str = "",
               ref_line: float | None = None, bands: list[tuple[float, float, str]] | None = None, w: int = 560, h: int = 260) -> str:
    """Multi-series line chart. ``bands`` are (x0, x1, css_class) background spans (used for before/during/after)."""
    pts = [p for s in series.values() for p in s]
    if not pts:
        return ""
    pad_l, pad_b, pad_t, pad_r = 52, 38, 14, 14
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys + ([ref_line] if ref_line is not None else [])), max(ys + ([ref_line] if ref_line is not None else []))
    if y1 == y0:
        y1 = y0 + 1
    span = y1 - y0
    y0, y1 = y0 - 0.08 * span, y1 + 0.08 * span
    px = lambda v: pad_l + (v - x0) / ((x1 - x0) or 1) * (w - pad_l - pad_r)      # noqa: E731
    py = lambda v: h - pad_b - (v - y0) / (y1 - y0) * (h - pad_b - pad_t)         # noqa: E731
    out = [_open(w, h, title, desc)]
    for a, b, cls in bands or []:
        out.append(f'<rect class="{cls}" x="{f(px(a))}" y="{pad_t}" width="{f(max(px(b) - px(a), 2))}" height="{h - pad_b - pad_t}"/>')
    out.append(f'<line class="axis" x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}"/><line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}"/>')
    for t in range(5):
        v = y0 + (y1 - y0) * t / 4
        out.append(f'<line class="grid" x1="{pad_l}" y1="{f(py(v))}" x2="{w - pad_r}" y2="{f(py(v))}"/><text class="tick" x="{pad_l - 6}" y="{f(py(v) + 3)}" text-anchor="end">{v:.3g}</text>')
    for xv in sorted(set(xs)):
        out.append(f'<text class="tick" x="{f(px(xv))}" y="{h - pad_b + 14}" text-anchor="middle">{xv:g}</text>')
    if ref_line is not None:
        out.append(f'<line class="diag" x1="{pad_l}" y1="{f(py(ref_line))}" x2="{w - pad_r}" y2="{f(py(ref_line))}"/>')
    for i, (name, s) in enumerate(series.items()):
        cls = f"series-{'abcd'[i % 4]}"
        d = " ".join(f"{'M' if j == 0 else 'L'}{f(px(x))} {f(py(y))}" for j, (x, y) in enumerate(s))
        out.append(f'<path class="line {cls}" d="{d}"/>')
        for x, y in s:
            out.append(f'<circle class="{cls}" cx="{f(px(x))}" cy="{f(py(y))}" r="3"><title>{esc(name)} @ {x:g}: {y:.4g}</title></circle>')
    out.append(f'<text class="label" x="{(pad_l + w - pad_r) // 2}" y="{h - 4}" text-anchor="middle">{esc(xlabel)}</text>'
               f'<text class="label" transform="rotate(-90 12 {h // 2})" x="12" y="{h // 2}" text-anchor="middle">{esc(ylabel)}</text>')
    for i, name in enumerate(series):
        out.append(f'<text class="legend series-{"abcd"[i % 4]}t" x="{pad_l + 8 + i * 150}" y="{pad_t + 10}">━ {esc(name)}</text>')
    out.append("</svg>")
    return "".join(out)


def hbar_chart(items: list[tuple[str, float]], title: str, desc: str, *, fmt: str = "{:.4g}", w: int = 480, row_h: int = 22) -> str:
    if not items:
        return ""
    label_w = 150
    mx = max(abs(v) for _, v in items) or 1.0
    h = row_h * len(items) + 20
    out = [_open(w, h, title, desc)]
    for i, (name, v) in enumerate(items):
        y = 10 + i * row_h
        bw = abs(v) / mx * (w - label_w - 70)
        out.append(f'<text class="tick" x="{label_w - 6}" y="{y + 13}" text-anchor="end">{esc(name)}</text>'
                   f'<rect class="series-a" x="{label_w}" y="{y + 2}" width="{f(bw)}" height="{row_h - 8}"><title>{esc(name)}: {fmt.format(v)}</title></rect>'
                   f'<text class="tick" x="{f(label_w + bw + 4)}" y="{y + 13}">{fmt.format(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def level_strip(rows: dict[str, list[str]], cols: list[str], title: str, desc: str, *, cell: int = 30, label_w: int = 150) -> str:
    """Status grid: one row per signal, one column per time window. Each cell shows an icon *and* a letter (not color alone)."""
    if not rows:
        return ""
    h = 24 + cell * len(rows) + 4
    w = label_w + cell * len(cols) + 8
    out = [_open(w, h, title, desc, "chart strip")]
    for j, c in enumerate(cols):
        out.append(f'<text class="tick" x="{label_w + j * cell + cell // 2}" y="14" text-anchor="middle">{esc(c)}</text>')
    for i, (name, levels) in enumerate(rows.items()):
        y = 22 + i * cell
        out.append(f'<text class="tick" x="{label_w - 6}" y="{y + cell // 2 + 3}" text-anchor="end">{esc(name)}</text>')
        for j, lv in enumerate(levels):
            lv = lv or "none"
            ic = ICON.get(lv, "·")
            letter = {"ok": "ok", "watch": "w", "alert": "A", "none": "n/a"}.get(lv, lv[:1])
            out.append(f'<g class="s-{lv}"><rect x="{label_w + j * cell + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" rx="3"><title>{esc(name)} / window {esc(cols[j])}: {lv}</title></rect>'
                       f'<text class="cell" x="{label_w + j * cell + cell // 2}" y="{y + cell // 2 + 4}" text-anchor="middle">{ic}{letter if letter != "ok" else ""}</text></g>')
    out.append("</svg>")
    return "".join(out)


def timeline_svg(events: list[dict], title: str, desc: str, w: int = 760) -> str:
    """Horizontal event timeline in sequence order (not to time scale). events: {label, kind, ok}."""
    if not events:
        return ""
    step = max(90, (w - 40) // max(len(events), 1))
    w = max(w, 40 + step * len(events))
    h = 96
    out = [_open(w, h, title, desc, "chart timeline"), f'<line class="axis" x1="20" y1="40" x2="{w - 20}" y2="40"/>']
    for i, e in enumerate(events):
        x = 30 + i * step
        cls = {"promote": "s-ok", "rollback": "s-watch", "reject": "s-alert", "canary_abort": "s-alert", "hold": "s-watch"}.get(e["kind"], "s-none")
        brk = e.get("broken")
        out.append(f'<g class="{"s-alert" if brk else cls}"><circle cx="{x}" cy="40" r="9"><title>#{e.get("seq", i + 1)} {esc(e["kind"])}</title></circle>'
                   f'<text class="cell" x="{x}" y="44" text-anchor="middle">{ICON.get({"promote": "ok", "reject": "alert", "canary_abort": "alert"}.get(e["kind"], "watch"), "·")}</text></g>'
                   f'<text class="tick" x="{x}" y="66" text-anchor="middle">{esc(e["kind"])}</text><text class="tick" x="{x}" y="80" text-anchor="middle">#{e.get("seq", i + 1)}{" BROKEN" if brk else ""}</text>')
    out.append("</svg>")
    return "".join(out)
