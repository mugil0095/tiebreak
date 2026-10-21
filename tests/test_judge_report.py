import pandas as pd

from tiebreak.demo import make_reviews
from tiebreak.notebook import compare
from tiebreak.report import render_report

N = 30
QUIET = dict(n_boot=300, concurrency=1, retries=0, sleep=lambda s: None)
JUDGE_TASK = {"type": "judge", "prompt_column": "prompt"}


def _frame(human=None):
    df = pd.DataFrame({"id": [f"q{i}" for i in range(N)], "prompt": [f"question number {i}" for i in range(N)]})
    if human is not None:
        df["human"] = human
    return df


def _models(judge_name="judge", judge_model="judge-model", cand_model=None):
    return [
        {"name": "long", "kind": "llm", "provider": "mock", "model": cand_model or "long",
         "params": {"answer_words": 200, "cost_per_call": 0.01}},
        {"name": "short", "kind": "llm", "provider": "mock", "model": "short",
         "params": {"answer_words": 5, "cost_per_call": 0.001}},
        {"name": judge_name, "kind": "llm", "provider": "mock", "model": judge_model, "role": "judge",
         "params": {"bias_by_length": True, "cost_per_call": 0.0005}},
    ]


def _judge_comparison(tmp_path, task=None, models=None, frame=None):
    return compare(task or JUDGE_TASK, models or _models(), frame if frame is not None else _frame(),
                   run_dir=tmp_path / "r", **QUIET)


def test_compare_runs_a_judge_task_from_a_dataframe(tmp_path):
    c = _judge_comparison(tmp_path)
    assert c.recommendation["winner"] == "long"
    assert set(c.table.index) == {"long", "short"}


def test_judge_section_is_present_for_judge_runs_only(tmp_path):
    judge_html = _judge_comparison(tmp_path).report_html()
    assert "<h2>Judge</h2>" in judge_html
    label_task = {"type": "classify", "label_column": "sentiment", "text_column": "review",
                  "labels": ["negative", "neutral", "positive"], "repeats": 1, "cv_folds": 3, "cv_repeats": 1}
    label_models = [{"name": "logreg", "kind": "ml", "estimator": "logistic_regression"}]
    label_html = compare(label_task, label_models, make_reviews(60, 0), run_dir=tmp_path / "l", n_boot=200).report_html()
    assert "<h2>Judge</h2>" not in label_html and "labelling mistakes" in label_html


def test_report_says_scores_come_from_a_model_and_shows_its_numbers(tmp_path):
    html = _judge_comparison(tmp_path).report_html()
    assert "judge model" in html and "not ground truth" in html
    assert "judge-model" in html  # the judge's model id
    assert "$0.0300" in html  # evaluation cost: 2 candidates x 30 answers x $0.0005 per judge call
    assert "labelling mistakes" not in html and "cross-validation" not in html


def test_self_judging_warning_is_shown_in_the_judge_section(tmp_path):
    html = _judge_comparison(tmp_path, models=_models(judge_model="long")).report_html()
    assert "also a candidate" in html


def test_human_agreement_is_shown_when_supplied(tmp_path):
    human = [str(1 + i % 5) for i in range(N)]
    models = _models()
    models[2]["params"] = {"cost_per_call": 0.0005}  # a judge whose scores vary
    html = _judge_comparison(tmp_path, task={**JUDGE_TASK, "human_score_column": "human"}, models=models,
                             frame=_frame(human)).report_html()
    assert "Spearman" in html and "30 rows" in html


def test_no_script_tags_and_hostile_names_are_escaped(tmp_path):
    hostile = "<script>alert(1)</script>"
    html = _judge_comparison(tmp_path, models=_models(judge_name=hostile, judge_model=hostile)).report_html()
    assert "<script" not in html.lower() and "&lt;script&gt;" in html


def test_render_report_tolerates_a_judge_block_with_missing_numbers():
    data = {"task": {"type": "judge", "metric": "mean_judge_score", "volume": 10, "n_scoring": 5, "scale": [1, 5]},
            "candidates": [], "recommendation": None, "judge": {"name": "j", "model": "m", "backend": "mock"}}
    assert "<h2>Judge</h2>" in render_report(data)
