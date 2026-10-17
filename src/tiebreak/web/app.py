"""Local web app: a thin FastAPI layer over the library. Localhost only; keys come from the environment, never the page."""
from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import load_models, load_pricing, load_task
from ..data import load_dataset
from ..demo import demo_files
from ..engine import SafetyError, _collect_secrets, is_hosted, new_run_dir, run_experiment
from ..report import render_report

RUN_ID = re.compile(r"run-[0-9]{8}-[0-9]{6}(-[0-9]+)?")
# Checked against the raw Host header (not a parsed URL: "evil.com:80@localhost" would parse as localhost).
LOCAL_HOST = re.compile(r"(localhost|127\.0\.0\.1|\[::1\])(:[0-9]{1,5})?", re.IGNORECASE)
STATIC = Path(__file__).parent / "static"
MAX_YAML_BYTES = 64 * 1024
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
SECURITY_HEADERS = {"X-Frame-Options": "SAMEORIGIN", "Content-Security-Policy": "frame-ancestors 'self'",
                    "X-Content-Type-Options": "nosniff"}


def _clean(message, models=()) -> str:
    """One line, secrets replaced, capped: the only form in which error text leaves the web layer."""
    text = str(message)
    for secret in _collect_secrets(list(models)):
        text = text.replace(secret, "[redacted]")
    return " ".join(text.split())[:300]


def _err(status: int, message, models=()) -> JSONResponse:
    return JSONResponse({"error": _clean(message, models)}, status_code=status, headers=SECURITY_HEADERS)


def _read_results(run_dir: Path):
    p = run_dir / "results.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except ValueError:  # half-written or corrupt results.json (JSON and unicode errors are ValueErrors)
        return None


def create_app(runs_dir="runs", max_upload_mb: float = 20) -> FastAPI:
    runs = Path(runs_dir)
    app = FastAPI(title="tiebreak", docs_url=None, redoc_url=None)
    lock = threading.Lock()                    # one run at a time
    state: dict[str, dict] = {}                # run id -> {"state", "error", "calls_total"}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.headers.get("host", "")
        if not LOCAL_HOST.fullmatch(host):
            return _err(403, "this app only answers requests addressed to localhost")
        if request.method not in SAFE_METHODS:
            # Cross-site defence: a web page open in the browser can POST a form to localhost.
            origin = request.headers.get("origin")
            if origin is not None and origin.lower() != f"http://{host}".lower():
                return _err(403, "cross-site request refused")
            if request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
                return _err(403, "cross-site request refused")
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    def run_dir_for(run_id: str) -> Path | None:
        # The id is validated before any filesystem access.
        if not RUN_ID.fullmatch(run_id):
            return None
        d = runs / run_id
        return d if d.is_dir() else None

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/demo")
    def demo():
        return demo_files()

    @app.get("/api/runs")
    def list_runs():
        out = []
        for d in sorted((p for p in runs.glob("run-*") if RUN_ID.fullmatch(p.name)), reverse=True) if runs.exists() else []:
            data = _read_results(d)
            rec = (data or {}).get("recommendation") or {}
            out.append({"id": d.name, "state": state.get(d.name, {}).get("state", "done" if data else "unknown"),
                        "winner": rec.get("winner"), "status": rec.get("status")})
        return out

    @app.post("/api/runs")
    def create_run(
        task_yaml: str = Form(...), models_yaml: str = Form(...), data: UploadFile = File(...),
        pricing_yaml: str = Form(""), allow_hosted: bool = Form(False), local_only: bool = Form(False),
        max_cost: float | None = Form(None, ge=0, allow_inf_nan=False),
        concurrency: int = Form(4, ge=1, le=16), n_boot: int = Form(10000, ge=100, le=1_000_000),
    ):
        content = data.file.read(int(max_upload_mb * 1024 * 1024) + 1)
        if len(content) > max_upload_mb * 1024 * 1024:
            return _err(413, f"the data file is larger than {max_upload_mb:g} MB")
        for name, text in (("task", task_yaml), ("models", models_yaml), ("pricing", pricing_yaml)):
            if len(text.encode("utf-8")) > MAX_YAML_BYTES:
                return _err(400, f"the {name} settings are larger than {MAX_YAML_BYTES // 1024} KB")
        if not lock.acquire(blocking=False):
            return _err(409, "a run is already in progress; wait for it to finish")
        run_dir = models = None
        try:
            run_dir = new_run_dir(runs)
            inputs = run_dir / "inputs"
            inputs.mkdir(parents=True)
            (inputs / "task.yaml").write_text(task_yaml, encoding="utf-8")
            (inputs / "models.yaml").write_text(models_yaml, encoding="utf-8")
            (inputs / "data.csv").write_bytes(content)
            task, models = load_task(inputs / "task.yaml"), load_models(inputs / "models.yaml")
            pricing = {}
            if pricing_yaml.strip():
                (inputs / "pricing.yaml").write_text(pricing_yaml, encoding="utf-8")
                pricing = load_pricing(inputs / "pricing.yaml")
            dataset = load_dataset(inputs / "data.csv", task)
            hosted = [m.name for m in models if is_hosted(m)]
            if hosted and local_only:
                raise SafetyError(f"local-only is on but these candidates are hosted: {hosted}")
            if hosted and not allow_hosted:
                raise SafetyError("hosted candidates send example text to third-party APIs; "
                                  "tick the consent box to allow it")
        except Exception as e:  # config, data and safety problems are the user's to fix: one line, no folder left
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)
            lock.release()
            return _err(400, e, models or ())

        run_id = run_dir.name
        n_llm = sum(1 for m in models if m.kind == "llm")
        state[run_id] = {"state": "running", "error": None, "calls_total": len(dataset) * task.repeats * n_llm}

        def work():
            try:
                run_experiment(task, models, dataset, run_dir, pricing=pricing, max_cost=max_cost,
                               concurrency=concurrency, allow_hosted=allow_hosted, local_only=local_only, n_boot=n_boot)
                state[run_id]["state"] = "done"
            except Exception as e:  # shown to the user as one line; the traceback is not exposed
                state[run_id].update(state="failed", error=_clean(e, models))
            finally:
                if state[run_id]["state"] == "running":  # a BaseException got past the handler
                    state[run_id].update(state="failed", error="the run stopped unexpectedly")
                lock.release()

        try:
            threading.Thread(target=work, daemon=True).start()
        except Exception as e:  # e.g. cannot start a thread: never leave the lock held
            state[run_id].update(state="failed", error=_clean(e, models))
            lock.release()
            return _err(500, f"could not start the run: {e}", models)
        return {"id": run_id}

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str):
        d = run_dir_for(run_id)
        if d is None:
            return _err(404, "no such run")
        st = state.get(run_id, {})
        data = _read_results(d)
        pred = d / "predictions.jsonl"
        done = 0
        if pred.exists():
            with pred.open(encoding="utf-8") as fh:
                done = sum(1 for line in fh if line.strip())
        rec = (data or {}).get("recommendation")
        return {"id": run_id, "state": st.get("state", "done" if data else "unknown"), "error": st.get("error"),
                "calls_done": done, "calls_total": st.get("calls_total", done), "recommendation": rec,
                "warnings": ((data or {}).get("warnings") or [])[:5]}

    @app.get("/api/runs/{run_id}/report", response_class=HTMLResponse)
    def run_report(run_id: str, redact: bool = False):
        d = run_dir_for(run_id)
        data = _read_results(d) if d else None
        if data is None:
            return _err(404, "no report yet")
        return HTMLResponse(render_report(data, redact))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")  # brand icons; the Host/Origin middleware still applies
    return app
