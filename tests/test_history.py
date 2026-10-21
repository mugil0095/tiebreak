import json

from typer.testing import CliRunner

from tiebreak.cli import app
from tiebreak.history import collect_history

runner = CliRunner()


def _write(runs, name, date, task_type, metric, candidates):
    d = runs / name
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "task": {"type": task_type, "metric": metric},
        "run_date": date,
        "candidates": candidates,
    }), encoding="utf-8")


def _cand(name, kind="ml", status="included", score=0.8, cost=0.01, p50=0.5):
    return {"name": name, "kind": kind, "status": status, "score": score,
            "total_cost_at_volume": cost, "p50_latency_s": p50}


def test_rows_come_from_every_valid_run_sorted_by_date(tmp_path):
    _write(tmp_path, "run-b", "2026-10-16", "binary", "macro_f1", [_cand("logreg", score=0.7)])
    _write(tmp_path, "run-a", "2026-10-15", "classify", "macro_f1",
           [_cand("logreg", score=0.845), _cand("gemini", kind="llm", score=0.825, cost=0.133, p50=3.2)])
    rows, skipped = collect_history(tmp_path)
    assert skipped == 0
    assert [(r.run, r.model) for r in rows] == [("run-a", "logreg"), ("run-a", "gemini"), ("run-b", "logreg")]
    first = rows[0]
    assert (first.date, first.task_type, first.metric, first.kind, first.status) == (
        "2026-10-15", "classify", "macro_f1", "ml", "included")
    assert (first.score, first.cost_at_volume, first.p50_latency_s) == (0.845, 0.01, 0.5)


def test_unreadable_runs_are_counted_not_fatal(tmp_path):
    _write(tmp_path, "good", "2026-10-15", "binary", "accuracy", [_cand("m")])
    (tmp_path / "no-results").mkdir()
    (tmp_path / "bad-json").mkdir()
    (tmp_path / "bad-json" / "results.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "wrong-shape").mkdir()
    (tmp_path / "wrong-shape" / "results.json").write_text('["a list"]', encoding="utf-8")
    rows, skipped = collect_history(tmp_path)
    assert len(rows) == 1 and skipped == 3


def test_missing_runs_dir_is_empty_not_an_error(tmp_path):
    assert collect_history(tmp_path / "nope") == ([], 0)


def test_null_scores_are_kept_as_none(tmp_path):
    _write(tmp_path, "r", "2026-10-15", "classify", "macro_f1",
           [_cand("eliminated", status="eliminated", score=None, cost=None, p50=None)])
    rows, _ = collect_history(tmp_path)
    assert rows[0].score is None and rows[0].cost_at_volume is None


def test_history_command_prints_table_and_skipped_note(tmp_path):
    _write(tmp_path, "r", "2026-10-15", "binary", "accuracy", [_cand("logreg")])
    (tmp_path / "broken").mkdir()
    r = runner.invoke(app, ["history", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0 and "logreg" in r.output and "skipped 1 unreadable run folder" in r.output


def test_history_command_csv(tmp_path):
    _write(tmp_path, "r", "2026-10-15", "binary", "accuracy", [_cand("logreg"), _cand("boosted")])
    r = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--csv"])
    lines = r.output.strip().splitlines()
    assert r.exit_code == 0 and len(lines) == 3
    assert lines[0].split(",")[:3] == ["run", "date", "task_type"]


def test_csv_stdout_stays_pure_when_runs_are_skipped(tmp_path):
    _write(tmp_path, "r", "2026-10-15", "binary", "accuracy", [_cand("logreg")])
    (tmp_path / "broken").mkdir()
    r = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--csv"])
    assert "skipped" not in r.stdout and len(r.stdout.strip().splitlines()) == 2

