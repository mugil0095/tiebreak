import pytest

import tiebreak
from tiebreak.data import DataError
from tiebreak.demo import make_reviews, write_demo
from tiebreak.engine import SafetyError, new_run_dir
from tiebreak.notebook import Comparison, compare

TASK = {"type": "classify", "label_column": "sentiment", "text_column": "review",
        "labels": ["negative", "neutral", "positive"], "repeats": 1, "cv_folds": 3, "cv_repeats": 1}
MODELS = [
    {"name": "logreg", "kind": "ml", "estimator": "logistic_regression"},
    {"name": "sim", "kind": "llm", "provider": "mock", "model": "sim",
     "params": {"accuracy": 0.9, "cost_per_call": 0.001}},
]


def frame(n=120):
    return make_reviews(n, 0)


def test_compare_runs_from_dicts_and_a_dataframe(tmp_path):
    c = compare(TASK, MODELS, frame(), run_dir=tmp_path / "r", n_boot=200)
    assert isinstance(c, Comparison)
    assert c.recommendation["winner"] in {"logreg", "sim"}
    assert set(c.table.index) == {"logreg", "sim"} and "score" in c.table.columns
    assert (tmp_path / "r" / "results.json").exists()


def test_compare_accepts_yaml_paths(tmp_path):
    write_demo(tmp_path)
    c = compare(tmp_path / "task.yaml", tmp_path / "models.yaml", tmp_path / "data.csv",
                run_dir=tmp_path / "r", n_boot=200)
    assert c.data["simulated"] is True and len(c.table) == 5


def test_repr_html_is_a_sandboxed_iframe_without_scripts(tmp_path):
    c = compare(TASK, MODELS, frame(), run_dir=tmp_path / "r", n_boot=200)
    h = c._repr_html_()
    assert h.startswith("<iframe sandbox") and "srcdoc=" in h and "<script" not in h.lower()


def test_save_report_and_redact(tmp_path):
    c = compare(TASK, MODELS, frame(), run_dir=tmp_path / "r", n_boot=200)
    out = c.save_report(tmp_path / "x.html", redact=True)
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert c.save_report().name == "report.html"


def test_safety_flags_pass_through(tmp_path):
    hosted = MODELS + [{"name": "h", "kind": "llm", "provider": "anthropic", "model": "x"}]
    with pytest.raises(SafetyError, match="yes-send-data"):
        compare(TASK, hosted, frame(), run_dir=tmp_path / "r", n_boot=200)


def test_dataframe_without_the_label_column_gives_the_clear_data_error(tmp_path):
    with pytest.raises(DataError, match="label column"):
        compare(TASK, MODELS, frame().drop(columns=["sentiment"]), run_dir=tmp_path / "r")


def test_default_run_dir_is_fresh_and_new_run_dir_avoids_collisions(tmp_path):
    a = new_run_dir(tmp_path)
    a.mkdir()
    assert new_run_dir(tmp_path) != a


def test_lazy_export():
    assert tiebreak.compare is compare
