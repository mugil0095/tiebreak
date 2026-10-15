from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from .svg import _num, frontier_svg

_env = Environment(loader=PackageLoader("tiebreak", "templates"),
                   autoescape=select_autoescape(["html", "j2"], default_for_string=True, default=True))
NA = "n/a"


def _f(v, spec: str, prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{format(v, spec)}{suffix}" if _num(v) else NA


def _row(c: dict) -> dict:
    star = "" if c.get("price_known", True) else "*"
    lat = (f"{c['p50_latency_s']:.3f}s / {c['p95_latency_s']:.3f}s"
           if _num(c.get("p50_latency_s")) and _num(c.get("p95_latency_s")) else NA)
    ci = (f"{c['ci_lo']:.3f} to {c['ci_hi']:.3f}"
          if _num(c.get("ci_lo")) and _num(c.get("ci_hi")) else NA)
    cost1k, total = _f(c.get("cost_per_1k"), ".4f", "$"), _f(c.get("total_cost_at_volume"), ".2f", "$")
    return {
        "name": c.get("name", ""), "kind": c.get("kind", ""), "status": c.get("status", ""),
        "tied": bool(c.get("equivalent")) and c.get("status") == "included", "note": c.get("note") or "",
        "score": _f(c.get("score"), ".3f"), "ci": ci,
        "cost_1k": cost1k + (star if cost1k != NA else ""), "total": total + (star if total != NA else ""),
        "latency": lat, "invalid": NA if not _num(c.get("invalid_rate")) else f"{c['invalid_rate'] * 100:.1f}%",
        "errors": NA if not _num(c.get("error_rate")) else f"{c['error_rate'] * 100:.1f}%",
    }


def render_report(data: dict, redact: bool = False) -> str:
    rec = data.get("recommendation")
    winners = set(rec.get("equivalent") or []) if rec else set()
    points = [
        {"name": c["name"], "x": c.get("total_cost_at_volume"), "y": c.get("score"), "lo": c.get("ci_lo"),
         "hi": c.get("ci_hi"), "highlight": c["name"] in winners}
        for c in data.get("candidates", []) if c.get("status") == "included"
    ]
    diag = data.get("diagnostics") or {}
    mislabels = [{"id": m.get("id"), "label": m.get("label"),
                  "pct": f"{(m.get('disagree_fraction') or 0) * 100:.0f}"} for m in diag.get("mislabels") or []]
    pairwise = [{"a": p.get("a"), "b": p.get("b"), "p": _f(p.get("mcnemar_p"), ".4f")} for p in data.get("pairwise") or []]
    return _env.get_template("report.html.j2").render(
        d=data, task=data.get("task") or {}, rec=rec, rows=[_row(c) for c in data.get("candidates", [])],
        chart=frontier_svg(points), redact=redact, mislabels=mislabels, pairwise=pairwise,
        n_disagree=len(diag.get("disagreement_ids") or []),
        warnings=data.get("warnings") or [], elimination_log=data.get("elimination_log") or [],
    )


def write_report(run_dir, redact: bool = False) -> Path:
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    out = run_dir / "report.html"
    out.write_text(render_report(data, redact), encoding="utf-8")
    return out
