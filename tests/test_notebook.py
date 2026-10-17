import subprocess
import sys

import pandas as pd
import pytest

import tiebreak
from tiebreak.config import ConfigError, TaskConfig
from tiebreak.data import DataError
from tiebreak.demo import make_reviews, write_demo
from tiebreak.engine import SafetyError, new_run_dir
from tiebreak.notebook import Comparison, _dataset, compare

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
    assert h.startswith("<iframe sandbox") and h.count("srcdoc=") == 1
    assert "allow-scripts" not in h.split("srcdoc=")[0]
    assert "<script" not in c.report_html().lower()


def test_repr_html_escapes_hostile_candidate_names(tmp_path):
    models = [dict(MODELS[0], name='a"b&c'), MODELS[1]]
    c = compare(TASK, models, frame(), run_dir=tmp_path / "r", n_boot=200)
    h = c._repr_html_()
    attr = h.split('srcdoc="', 1)[1].split('" style=', 1)[0]
    assert '"' not in attr and "a&quot;b&amp;c" in attr.replace("&amp;quot;", "&quot;").replace("&amp;amp;", "&amp;")


def test_dataframe_round_trip_keeps_none_labels_and_leading_zero_ids():
    task = TaskConfig(type="classify", label_column="y", text_column="t", labels=["None", "yes"])
    df = pd.DataFrame({"id": ["007", "008", "009", "010"], "t": list("abcd"), "y": ["None", "yes", "None", "yes"]})
    ds = _dataset(df, task)
    assert ds.ids == ["007", "008", "009", "010"] and list(ds.y) == ["None", "yes", "None", "yes"]


def test_pricing_accepts_yaml_shape_and_pairs(tmp_path):
    for i, price in enumerate([{"input_per_1k": 0.5, "output_per_1k": 1.5}, [0.5, 1.5]]):
        c = compare(TASK, MODELS, frame(), pricing={"sim": price}, run_dir=tmp_path / f"r{i}", n_boot=200)
        used = c.data["prices_used"][0]
        assert (used["input_per_1k"], used["output_per_1k"]) == (0.5, 1.5)


@pytest.mark.parametrize("bad", [{"m": [0.1]}, {"m": {"input_per_1k": 1}}, {"m": 3}, {"m": {"input_per_1k": "x", "output_per_1k": 1}}])
def test_malformed_pricing_names_the_model(bad):
    from tiebreak.notebook import _pricing
    with pytest.raises(ConfigError, match="'m'"):
        _pricing(bad)


def test_import_tiebreak_stays_light():
    code = ("import sys, tiebreak; "
            "bad = [m for m in ('tiebreak.notebook', 'pandas', 'fastapi') if m in sys.modules]; "
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


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
    assert "compare" in dir(tiebreak)
