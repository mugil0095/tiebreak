from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
from pydantic import ValidationError

from .candidates.ml import ESTIMATORS
from .config import load_models, load_pricing, load_task
from .data import DataError, load_dataset
from .demo import DEMO_FILES, write_demo
from .engine import SafetyError, run_experiment
from .planner import estimate
from .protocol import make_split
from .report import write_report

app = typer.Typer(add_completion=False, help="Pick the cheapest model that is statistically tied with the best.")
# Expected user-facing problems: one clean line and exit code 2, no traceback. (ConfigError is a ValueError.)
CLEAN_ERRORS = (SafetyError, DataError, ValueError, FileNotFoundError, ImportError, ValidationError)
BOOST_ESTIMATORS = {"xgboost", "lightgbm"}
MAX_LISTED = 20

_TASK_TEMPLATE = """# What you want to compare models on. Edit, then run: tiebreak estimate / tiebreak run
type: classify            # classify | binary | extract_exact | regress
label_column: label       # column in your data holding the correct answer
text_column: text         # column with the text to classify (omit for tabular data)
labels: [a, b, c]         # the allowed labels (classify/binary)
margin: 0.02              # "tied" means within this much of the best score
volume: 10000             # how many items you will process per period, for cost projection
repeats: 1                # times each LLM call is repeated (1 = cheapest)
"""

_MODELS_TEMPLATE = """# One entry per candidate. Hosted providers send your example text to a third party:
# tiebreak refuses to run them unless you pass --yes-send-data.
- {name: logreg, kind: ml, estimator: logistic_regression}
# - {name: my-hosted, kind: llm, provider: <provider>, model: "<model id from your account>"}
# - {name: local, kind: llm, provider: ollama, model: "<local model tag>"}
# - {name: trial, kind: llm, provider: mock, model: trial, params: {accuracy: 0.9, cost_per_call: 0.001}}  # SIMULATED
"""


def _fail(e: Exception):
    if isinstance(e, ValidationError):
        msg = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
    elif isinstance(e, FileNotFoundError) and e.filename:
        msg = f"file not found: {e.filename}"
    else:
        msg = " ".join(str(e).split())
    typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(code=2)


@app.command()
def init(demo: bool = typer.Option(False, "--demo", help="Write a synthetic demo dataset and config (SIMULATED models)."),
         dir: Path = typer.Option(Path("."), "--dir", help="Where to write files.")):
    """Write starter files: a runnable demo (--demo) or commented templates."""
    names = DEMO_FILES if demo else ("task.yaml", "models.yaml")
    existing = [n for n in names if (dir / n).exists()]
    if existing:
        _fail(ValueError(f"{', '.join(existing)} already exist in {dir}; choose another --dir or remove them"))
    if demo:
        write_demo(dir)
        typer.echo(f"Wrote demo files to {dir} (the models in it are SIMULATED).")
        typer.echo("Next: tiebreak run --task task.yaml --models models.yaml --data data.csv")
        return
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "task.yaml").write_text(_TASK_TEMPLATE, encoding="utf-8")
    (dir / "models.yaml").write_text(_MODELS_TEMPLATE, encoding="utf-8")
    typer.echo(f"Wrote task.yaml and models.yaml templates to {dir}. Edit them, add your data file, then:")
    typer.echo("  tiebreak estimate --task task.yaml --models models.yaml --data <your data>")


@app.command()
def models():
    """List built-in ML estimators and LLM providers."""
    typer.echo("Built-in ML estimators:")
    for name in ESTIMATORS:
        typer.echo(f"  {name}" + ("   (needs the 'boost' extra: pip install tiebreak[boost])" if name in BOOST_ESTIMATORS else ""))
    typer.echo("LLM candidates use provider/model ids from models.yaml: any LiteLLM provider (needs the 'llm' extra), "
               "ollama, openai_compatible, or mock (SIMULATED, no API calls).")


def _load(task, models_path, data, pricing):
    t = load_task(task)
    return t, load_models(models_path), load_dataset(data, t), load_pricing(pricing)


@app.command("estimate")
def estimate_cmd(task: Path = typer.Option(..., "--task"), models_path: Path = typer.Option(..., "--models"),
                 data: Path = typer.Option(..., "--data"), pricing: Path = typer.Option(None, "--pricing")):
    """Dry run: how many calls and roughly what cost. It never calls any model."""
    try:
        t, m, ds, p = _load(task, models_path, data, pricing)
        est = estimate(t, m, make_split(ds, t), p)
    except CLEAN_ERRORS as e:
        _fail(e)
    typer.echo("Estimate only: this never calls any model and sends no data anywhere.")
    for name, n in est.calls.items():
        c = est.cost[name]
        src = est.sources.get(name)
        typer.echo(f"{name}: {n} calls, est. cost " + ("unknown" if c is None else f"${c:.4f}") + (f" ({src})" if src else ""))
    if not est.calls:
        typer.echo("No LLM candidates: nothing to estimate (ML candidates make no paid calls).")
    typer.echo(f"Total estimated cost (priced models only): ${est.total:.4f}")
    for note in est.notes:
        typer.echo(f"note: {note}")


def _new_run_dir(out: Path) -> Path:
    base = f"run-{datetime.now():%Y%m%d-%H%M%S}"
    cand, k = out / base, 1
    while cand.exists():
        k += 1
        cand = out / f"{base}-{k}"
    return cand


@app.command()
def run(task: Path = typer.Option(..., "--task"), models_path: Path = typer.Option(..., "--models"),
        data: Path = typer.Option(..., "--data"), pricing: Path = typer.Option(None, "--pricing"),
        out: Path = typer.Option(Path("runs"), "--out"), max_cost: float = typer.Option(None, "--max-cost", min=0),
        concurrency: int = typer.Option(4, "--concurrency", min=1),
        yes_send_data: bool = typer.Option(False, "--yes-send-data", help="Allow hosted providers to receive example text."),
        local_only: bool = typer.Option(False, "--local-only", help="Refuse to call any hosted provider."),
        resume: Path = typer.Option(None, "--resume", help="Existing run folder to continue."),
        boot: int = typer.Option(10_000, "--boot", hidden=True)):
    """Run all candidates and write a report."""
    try:
        if resume is not None and not resume.is_dir():
            raise FileNotFoundError(f"--resume folder not found: {resume}")
        t, m, ds, p = _load(task, models_path, data, pricing)
        run_dir = resume if resume is not None else _new_run_dir(out)
        res = run_experiment(t, m, ds, run_dir, pricing=p, max_cost=max_cost, concurrency=concurrency,
                             allow_hosted=yes_send_data, local_only=local_only, n_boot=boot)
        report = write_report(run_dir)
    except CLEAN_ERRORS as e:
        _fail(e)
    _print_summary(res.data, report)


def _print_summary(data: dict, report) -> None:
    """Only candidate names, scores, the recommendation and a few warnings; never example text."""
    rec = data["recommendation"]
    if rec and rec.get("status") == "provisional":
        typer.echo(f"Result: PROVISIONAL ({rec['winner']}). {rec['reason'].removeprefix('Provisional: ')}")
    elif rec and rec.get("inconclusive"):
        typer.echo(f"Result: INCONCLUSIVE ({rec['status']}). {rec['reason']}")
    elif rec:
        typer.echo(f"Recommendation: {rec['winner']} ({rec['status']}). {rec['reason']}")
    else:
        typer.echo("No recommendation: nothing completed the evaluation.")
    if data.get("incomplete"):
        typer.echo("The run is incomplete (cost cap reached). Use --resume with a higher --max-cost.")
    warnings = list(data.get("warnings") or [])
    shown = warnings[:3]
    sim = [w for w in warnings if w.startswith("SIMULATED")]
    if sim and sim[0] not in shown:
        shown = [sim[0]] + shown[:2]
    shown.sort(key=lambda w: not w.startswith("SIMULATED"))
    for w in shown:
        typer.echo(f"warning: {w}")
    if len(warnings) > len(shown):
        typer.echo(f"... and {len(warnings) - len(shown)} more warnings in the report")
    typer.echo(f"Report: {Path(report).resolve()}")


def _read_results(run_dir: Path) -> dict:
    path = run_dir / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"results.json not found in {run_dir}; is this a finished run folder?")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"{path} is not valid JSON") from e


@app.command("report")
def report_cmd(run_dir: Path, redact: bool = typer.Option(False, "--redact", help="Hide example ids.")):
    """Regenerate report.html from a finished run (no model calls)."""
    try:
        _read_results(run_dir)
        typer.echo(f"Wrote {Path(write_report(run_dir, redact)).resolve()}")
    except CLEAN_ERRORS as e:
        _fail(e)


@app.command()
def diagnose(run_dir: Path):
    """Print diagnostics from a finished run."""
    try:
        d = _read_results(run_dir).get("diagnostics")
        if not isinstance(d, dict):
            raise ValueError(f"{run_dir / 'results.json'} has no diagnostics section")
    except CLEAN_ERRORS as e:
        _fail(e)
    mis = d.get("mislabels") or []
    typer.echo(f"Suspected labelling mistakes: {len(mis)}")
    for m in mis[:MAX_LISTED]:
        typer.echo(f"  {m['id']}: labelled '{m['label']}', {m['disagree_fraction']:.0%} of candidates disagree")
    if len(mis) > MAX_LISTED:
        typer.echo(f"  ... and {len(mis) - MAX_LISTED} more")
    typer.echo(f"Examples where candidates disagree: {len(d.get('disagreement_ids') or [])}")
    typer.echo(f"Class balance: {d.get('class_balance', {})}")
