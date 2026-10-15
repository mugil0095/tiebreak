from __future__ import annotations

from html import escape
from math import isfinite, log10

W, H, PAD = 640, 320, 48
COST_FLOOR = 1e-6  # dollars; zero or negative costs (free local models) are drawn here


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v)


def frontier_svg(points: list[dict]) -> str:
    """Score vs total cost (log scale) with confidence-interval bars. Inline SVG, no scripts.

    Points without a finite score or cost are left out. Costs <= 0 are clamped to a small
    positive value so the log scale works, and the axis label says so.
    """
    pts = [p for p in points if _num(p.get("y")) and _num(p.get("x"))]
    if not pts:
        return ""
    clamped = any(p["x"] < COST_FLOOR for p in pts)
    lx = lambda v: log10(max(v, COST_FLOOR))
    xs = [lx(p["x"]) for p in pts]
    x0, x1 = min(xs), max(xs)
    ys = [p["y"] for p in pts]
    for p in pts:
        ys += [v for v in (p.get("lo"), p.get("hi")) if _num(v)]
    y0, y1 = min(ys), max(ys)
    sx = lambda v: PAD + (0.5 if x1 == x0 else (lx(v) - x0) / (x1 - x0)) * (W - 2 * PAD)
    sy = lambda v: H - PAD - (0.5 if y1 == y0 else (v - y0) / (y1 - y0)) * (H - 2 * PAD)
    xlabel = "total cost at expected volume (log scale, $)"
    if clamped:
        xlabel += f"; costs at or below $0 are clamped to ${COST_FLOOR:g}"
    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
            f'aria-label="Score versus cost">'),
           f'<line x1="{PAD}" y1="{H - PAD}" x2="{W - PAD}" y2="{H - PAD}" stroke="#888"/>',
           f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H - PAD}" stroke="#888"/>',
           f'<text x="{W / 2}" y="{H - 10}" text-anchor="middle" font-size="12" fill="#555">{escape(xlabel)}</text>',
           (f'<text x="14" y="{H / 2}" font-size="12" fill="#555" transform="rotate(-90 14 {H / 2})" '
            f'text-anchor="middle">score</text>')]
    for p in pts:
        cx, cy = sx(p["x"]), sy(p["y"])
        color = "#0b7a5f" if p.get("highlight") else "#5b6b8c"
        if _num(p.get("lo")) and _num(p.get("hi")):
            out.append(f'<line x1="{cx:.1f}" y1="{sy(p["lo"]):.1f}" x2="{cx:.1f}" y2="{sy(p["hi"]):.1f}" '
                       f'stroke="{color}" stroke-width="2"/>')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}"/>')
        out.append(f'<text x="{cx + 9:.1f}" y="{cy - 8:.1f}" font-size="12" fill="#222">{escape(str(p["name"]))}</text>')
    out.append("</svg>")
    return "".join(out)
