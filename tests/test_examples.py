import json
from pathlib import Path

from typer.testing import CliRunner

from tiebreak.cli import app

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
runner = CliRunner()


def test_the_offline_judge_example_runs_end_to_end(tmp_path):
    d = EXAMPLES / "judge"
    r = runner.invoke(app, ["run", "--task", str(d / "task.yaml"), "--models", str(d / "models.yaml"),
                            "--data", str(d / "prompts.csv"), "--out", str(tmp_path), "--boot", "300"])
    assert r.exit_code == 0, r.output
    run = next(tmp_path.glob("run-*"))
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert data["task"]["type"] == "judge" and data["simulated"] is True
    assert data["recommendation"]["winner"] == "long-answers"
    assert data["judge"]["evaluation_cost"] > 0
    assert (run / "report.html").exists() and (run / "judgments.jsonl").exists()


def test_example_models_files_declare_exactly_one_judge():
    import yaml

    for name in ("models.yaml", "models-ollama.yaml", "models-jev.yaml"):
        entries = yaml.safe_load((EXAMPLES / "judge" / name).read_text(encoding="utf-8"))
        assert sum(1 for e in entries if e.get("role") == "judge") == 1, name
