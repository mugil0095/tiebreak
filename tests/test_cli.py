import json

import yaml
from typer.testing import CliRunner

from tiebreak.cli import app

runner = CliRunner()


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


def test_inconclusive_is_said_in_first_line(tmp_path):
    work = _demo(tmp_path)
    (work / "models.yaml").write_text(
        "- {name: a, kind: llm, provider: mock, model: a, params: {accuracy: 0.60, cost_per_call: 0.01}}\n"
        "- {name: b, kind: llm, provider: mock, model: b, params: {accuracy: 0.60, cost_per_call: 0.02}}\n")
    (work / "task.yaml").write_text(
        "type: classify\nlabel_column: sentiment\ntext_column: review\nlabels: [negative, neutral, positive]\n"
        "margin: 0.001\nvolume: 1000\nrepeats: 1\n")
    r = _run(work)
    assert r.exit_code == 0, r.output
    res = json.loads(next((work / "runs").iterdir()).joinpath("results.json").read_text())
    if res["recommendation"]["inconclusive"]:
        assert "inconclusive" in r.output.splitlines()[0].lower()


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
    assert "--models" in runner.invoke(app, ["run", "--help"]).output
