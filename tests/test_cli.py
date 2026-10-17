import json
import re
import subprocess
import sys

import pytest
import yaml
from typer.testing import CliRunner

from tiebreak.cli import _print_summary, app

runner = CliRunner()


def _plain(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)  # rich adds colour codes on CI


def _demo(tmp_path, name="my demo"):
    work = tmp_path / name  # a path with a space on purpose
    r = runner.invoke(app, ["init", "--demo", "--dir", str(work)])
    assert r.exit_code == 0, r.output
    return work


def _run(work, *extra, out=None):
    return runner.invoke(app, ["run", "--task", str(work / "task.yaml"), "--models", str(work / "models.yaml"),
                               "--data", str(work / "data.csv"), "--out", str(out or work / "runs"),
                               "--boot", "200", *extra])


def test_init_demo_then_run_produces_a_report(tmp_path):
    work = _demo(tmp_path)
    assert (work / "data.csv").exists() and (work / "task.yaml").exists() and (work / "models.yaml").exists()
    out = work / "runs"
    r = _run(work, out=out)
    assert r.exit_code == 0, r.output
    run_dir = next(out.iterdir())
    assert run_dir.name.startswith("run-")
    assert (run_dir / "report.html").exists() and (run_dir / "results.json").exists()
    assert "recommend" in r.output.lower()
    assert str((run_dir / "report.html").resolve()) in r.output.replace("\n", "")
    assert "SIMULATED" in r.output
    assert json.loads((run_dir / "results.json").read_text())["simulated"] is True


def test_demo_files_are_valid_and_say_simulated(tmp_path):
    work = _demo(tmp_path)
    models = yaml.safe_load((work / "models.yaml").read_text())
    assert {m["provider"] for m in models if m["kind"] == "llm"} == {"mock"}
    assert "SIMULATED" in (work / "models.yaml").read_text()
    assert "SIMULATED" in (work / "README.txt").read_text()
    assert "0.05" in (work / "README.txt").read_text() and "300 examples cannot confirm" in (work / "README.txt").read_text()


def test_two_runs_get_distinct_folders(tmp_path):
    work = _demo(tmp_path)
    assert _run(work).exit_code == 0 and _run(work).exit_code == 0
    assert len(list((work / "runs").iterdir())) == 2


def test_resume_reuses_the_folder(tmp_path):
    work = _demo(tmp_path)
    assert _run(work).exit_code == 0
    run_dir = next((work / "runs").iterdir())
    r = _run(work, "--resume", str(run_dir))
    assert r.exit_code == 0, r.output
    assert len(list((work / "runs").iterdir())) == 1


def _data(inconclusive, warnings=()):
    rec = {"status": "inconclusive" if inconclusive else "recommended", "winner": None if inconclusive else "w",
           "inconclusive": inconclusive, "reason": "because"}
    return {"recommendation": rec, "warnings": list(warnings), "incomplete": False}


def test_inconclusive_is_said_in_first_line(capsys):
    _print_summary(_data(True), "r.html")
    assert "INCONCLUSIVE" in capsys.readouterr().out.splitlines()[0]
    _print_summary(_data(False), "r.html")
    assert "INCONCLUSIVE" not in capsys.readouterr().out


def test_simulated_warning_always_shown_first(capsys):
    ws = ["w1", "w2", "w3", "w4", "SIMULATED: fake"]
    _print_summary(_data(False, ws), "r.html")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("warning:")]
    assert lines[0] == "warning: SIMULATED: fake" and len(lines) == 3


def test_option_bounds(tmp_path):
    work = _demo(tmp_path)
    assert _run(work, "--max-cost", "-1").exit_code != 0
    assert _run(work, "--concurrency", "0").exit_code != 0


def test_python_dash_m_tiebreak():
    r = subprocess.run([sys.executable, "-m", "tiebreak", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "tiebreak" in (r.stdout + r.stderr).lower()


def test_demo_shows_cheap_model_tying_expensive(tmp_path):
    work = _demo(tmp_path)
    assert _run(work).exit_code == 0
    res = json.loads(next((work / "runs").iterdir()).joinpath("results.json").read_text())
    rows = {c["name"]: c for c in res["candidates"]}
    assert rows["sim-small"]["equivalent"] and rows["sim-large"]["equivalent"]
    assert not rows["sim-weak"]["equivalent"]
    assert res["recommendation"]["winner"] == "sim-small"


def test_report_command_regenerates_and_redacts(tmp_path):
    work = _demo(tmp_path, "w")
    _run(work)
    run_dir = next((work / "runs").iterdir())
    (run_dir / "report.html").unlink()
    r = runner.invoke(app, ["report", str(run_dir), "--redact"])
    assert r.exit_code == 0 and (run_dir / "report.html").exists()
    r = runner.invoke(app, ["diagnose", str(run_dir)])
    assert r.exit_code == 0 and "disagree" in r.output.lower()


def test_diagnose_caps_mislabel_list(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mis = [{"id": f"id{i}", "label": "x", "disagree_fraction": 1.0} for i in range(30)]
    (run_dir / "results.json").write_text(json.dumps(
        {"diagnostics": {"mislabels": mis, "disagreement_ids": ["a"], "class_balance": {"x": 1}}}))
    r = runner.invoke(app, ["diagnose", str(run_dir)])
    assert r.exit_code == 0
    assert "id19" in r.output and "id20" not in r.output and "and 10 more" in r.output


def test_report_and_diagnose_fail_cleanly_on_missing_folder(tmp_path):
    for cmd in (["report", str(tmp_path / "nope")], ["diagnose", str(tmp_path / "nope")]):
        r = runner.invoke(app, cmd)
        assert r.exit_code == 2 and "Traceback" not in r.output and "results.json" in r.output
    (tmp_path / "empty").mkdir()
    r = runner.invoke(app, ["diagnose", str(tmp_path / "empty")])
    assert r.exit_code == 2 and "results.json" in r.output


def test_estimate_prints_call_counts_and_never_calls_models(tmp_path):
    work = _demo(tmp_path, "w")
    r = runner.invoke(app, ["estimate", "--task", str(work / "task.yaml"), "--models", str(work / "models.yaml"),
                            "--data", str(work / "data.csv")])
    assert r.exit_code == 0 and "calls" in r.output.lower()
    assert "sim-large" in r.output and "never calls" in r.output.lower()


def test_hosted_candidate_is_refused_without_consent(tmp_path):
    work = _demo(tmp_path, "w")
    (work / "hosted.yaml").write_text("- {name: h, kind: llm, provider: anthropic, model: some-model}\n")
    r = runner.invoke(app, ["run", "--task", str(work / "task.yaml"), "--models", str(work / "hosted.yaml"),
                            "--data", str(work / "data.csv"), "--out", str(work / "runs")])
    assert r.exit_code == 2 and "yes-send-data" in r.output and "Traceback" not in r.output
    r = runner.invoke(app, ["run", "--task", str(work / "task.yaml"), "--models", str(work / "hosted.yaml"),
                            "--data", str(work / "data.csv"), "--out", str(work / "runs"),
                            "--yes-send-data", "--local-only"])
    assert r.exit_code == 2 and "local-only" in r.output


def test_bad_data_gives_a_clean_message(tmp_path):
    work = _demo(tmp_path, "w")
    (work / "bad.csv").write_text("review,sentiment\nhello,\n")
    r = runner.invoke(app, ["run", "--task", str(work / "task.yaml"), "--models", str(work / "models.yaml"),
                            "--data", str(work / "bad.csv"), "--out", str(work / "runs")])
    assert r.exit_code == 2 and "no label" in r.output and "Traceback" not in r.output


def test_config_problems_are_clean_one_liners(tmp_path):
    work = _demo(tmp_path, "w")
    (work / "empty.yaml").write_text("")
    (work / "nolabel.yaml").write_text("type: classify\nlabels: [a, b]\n")
    for task, needle in (("empty.yaml", "empty"), ("nolabel.yaml", "label_column"), ("missing.yaml", "missing.yaml")):
        r = runner.invoke(app, ["estimate", "--task", str(work / task), "--models", str(work / "models.yaml"),
                                "--data", str(work / "data.csv")])
        assert r.exit_code == 2 and needle in r.output and "Traceback" not in r.output, (task, r.output)


def test_summary_never_prints_example_text(tmp_path):
    work = _demo(tmp_path, "w")
    r = _run(work)
    first_review = (work / "data.csv").read_text().splitlines()[1].split(",")[1]
    assert first_review not in r.output


def test_models_lists_builtin_estimators():
    r = runner.invoke(app, ["models"])
    assert r.exit_code == 0 and "logistic_regression" in r.output
    assert "boost" in r.output and "xgboost" in r.output and "mock" in r.output


def test_init_without_demo_writes_templates(tmp_path):
    d = tmp_path / "new proj"
    r = runner.invoke(app, ["init", "--dir", str(d)])
    assert r.exit_code == 0, r.output
    assert (d / "task.yaml").exists() and (d / "models.yaml").exists() and not (d / "data.csv").exists()
    assert yaml.safe_load((d / "task.yaml").read_text())["type"] == "classify"
    assert isinstance(yaml.safe_load((d / "models.yaml").read_text()), list)
    (d / "task.yaml").write_text("keep: me\n")
    r = runner.invoke(app, ["init", "--dir", str(d)])
    assert r.exit_code == 2 and (d / "task.yaml").read_text() == "keep: me\n"


def test_help_works():
    for args in (["--help"], ["run", "--help"], ["estimate", "--help"]):
        r = runner.invoke(app, args)
        assert r.exit_code == 0, (args, r.output)
    assert "--models" in _plain(runner.invoke(app, ["run", "--help"]).output)


def test_provisional_is_said_in_first_line(capsys):
    d = _data(False)
    d["recommendation"].update(status="provisional", reason="Provisional: left out: big (incomplete: provider errors). w is best.")
    _print_summary(d, "r.html")
    first = capsys.readouterr().out.splitlines()[0]
    assert "PROVISIONAL" in first and "(recommended)" not in first and "w" in first


def test_estimate_shows_simulated_cost_and_source_for_mock_models(tmp_path):
    work = _demo(tmp_path, "w")
    r = runner.invoke(app, ["estimate", "--task", str(work / "task.yaml"), "--models", str(work / "models.yaml"),
                            "--data", str(work / "data.csv")])
    line = next(ln for ln in r.output.splitlines() if ln.startswith("sim-large"))
    assert "(simulated)" in line and "unknown" not in line and "$3.0000" in line  # 300 calls x $0.01


def test_run_prints_the_run_folder_first_and_demo_hint_uses_the_real_dir(tmp_path):
    work = tmp_path / "proj dir"
    r = runner.invoke(app, ["init", "--demo", "--dir", str(work)])
    assert str(work / "task.yaml") in r.output.replace("\n", "") and "--data" in r.output
    r = _run(work)
    first = r.output.splitlines()[0]
    run_dir = next((work / "runs").iterdir())
    assert first.startswith("run folder: ") and str(run_dir.resolve()) in r.output.replace("\n", "")


def test_ctrl_c_prints_a_resume_hint_and_exits_130_without_a_traceback(tmp_path, monkeypatch):
    work = _demo(tmp_path)
    import tiebreak.cli as cli

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_experiment", boom)
    r = _run(work)
    assert r.exit_code == 130 and "Traceback" not in r.output
    assert "--resume" in r.output and "tiebreak run" in r.output and "interrupted" in r.output.lower()


def test_serve_help_lists_the_options():
    r = runner.invoke(app, ["serve", "--help"])
    assert r.exit_code == 0 and "--port" in _plain(r.output) and "--runs-dir" in _plain(r.output)


def test_serve_without_web_extra_gives_a_clean_message(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] in {"uvicorn", "fastapi"}:
            raise ImportError("no module")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    r = runner.invoke(app, ["serve"])
    assert r.exit_code == 2 and 'tiebreak[web]' in r.output


def test_serve_binds_loopback_only_and_passes_options(monkeypatch, tmp_path):
    uvicorn = pytest.importorskip("uvicorn")  # needs the web extra
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda application, **kw: seen.update(app=application, **kw))
    r = runner.invoke(app, ["serve", "--port", "9123", "--runs-dir", str(tmp_path / "r")])
    assert r.exit_code == 0, r.output
    assert seen["host"] == "127.0.0.1" and seen["port"] == 9123 and "127.0.0.1:9123" in r.output
    assert seen["app"].title == "tiebreak"
