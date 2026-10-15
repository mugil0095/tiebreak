import copy
import json
import re

from tiebreak.config import ModelSpec, TaskConfig
from tiebreak.data import Dataset
from tiebreak.demo import make_reviews
from tiebreak.engine import run_experiment
from tiebreak.report import render_report, write_report
from tiebreak.svg import frontier_svg

DATA = {
    "task": {"type": "classify", "metric": "macro_f1", "margin": 0.02, "volume": 10000, "n_scoring": 120,
             "shots_pool": 0, "repeats": 1, "cv_folds": 3, "cv_repeats": 1},
    "candidates": [
        {"name": "logreg", "kind": "ml", "status": "included", "note": "", "score": 0.95, "ci_lo": 0.9, "ci_hi": 0.98,
         "cost_per_1k": 0.0001, "total_cost_at_volume": 0.01, "price_known": True, "p50_latency_s": 0.001,
         "p95_latency_s": 0.002, "invalid_rate": 0.0, "error_rate": 0.0, "repeat_disagreement": 0.0, "equivalent": True},
        {"name": "big", "kind": "llm", "status": "included", "note": "", "score": 0.96, "ci_lo": 0.92, "ci_hi": 0.99,
         "cost_per_1k": 50.0, "total_cost_at_volume": 500.0, "price_known": True, "p50_latency_s": 0.4,
         "p95_latency_s": 0.9, "invalid_rate": 0.01, "error_rate": 0.0, "repeat_disagreement": 0.02, "equivalent": True},
        {"name": "bad", "kind": "llm", "status": "eliminated", "note": "eliminated early after 30 of 120 examples",
         "score": 0.4, "ci_lo": None, "ci_hi": None, "cost_per_1k": 5.0, "total_cost_at_volume": 50.0,
         "price_known": True, "p50_latency_s": 0.2, "p95_latency_s": 0.3, "invalid_rate": 0.0, "error_rate": 0.0,
         "repeat_disagreement": 0.0, "equivalent": False},
    ],
    "recommendation": {"status": "recommended", "winner": "logreg", "best": "big", "equivalent": ["big", "logreg"],
                       "inconclusive": [], "extra_examples": {}, "reason": "logreg is confirmed within 0.02 of the best."},
    "diagnostics": {"mislabels": [{"id": "r7", "label": "positive", "disagree_fraction": 1.0}],
                    "disagreement_ids": ["r1", "r2"], "class_balance": {"a": 40, "b": 80}},
    "pairwise": [{"a": "logreg", "b": "big", "mcnemar_p": 0.5}],
    "warnings": ["SIMULATED: fake models", "Only 20 examples."],
    "incomplete": False, "simulated": True, "elimination_log": ["bad: dropped"],
}


def test_report_is_self_contained_and_shows_the_decision():
    html = render_report(DATA)
    assert "logreg" in html and "confirmed within 0.02" in html and "SIMULATED" in html
    assert "<script" not in html.lower()
    assert not re.search(r'(src|href)="https?://', html)
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "<svg" in html and "r7" in html
    assert "conservative" in html


def test_redact_hides_example_ids():
    html = render_report(DATA, redact=True)
    assert "r7" not in html and "redacted" in html.lower()


def test_inconclusive_and_no_recommendation_render():
    rec_html = render_report(DATA)
    assert "Recommendation:" in rec_html and "Safe choice" not in rec_html
    assert '<div class="card rec">' in rec_html
    d = json.loads(json.dumps(DATA))
    d["recommendation"]["status"] = "inconclusive"
    d["recommendation"]["inconclusive"] = ["logreg"]
    html = render_report(d)
    assert "Safe choice" in html and "Recommendation:" not in html
    assert "Not enough data" in html
    assert '<div class="card rec">' not in html  # inconclusive uses the warning style, not the green one
    d["recommendation"] = None
    assert "nothing to recommend" in render_report(d).lower()


def test_mislabel_list_is_capped_and_redaction_still_applies():
    d = copy.deepcopy(DATA)
    d["diagnostics"]["mislabels"] = [{"id": f"id{i}", "label": "a", "disagree_fraction": 1.0} for i in range(40)]
    html = render_report(d)
    assert "id24 " in html and "id25 " not in html
    assert "and 15 more (see results.json)" in html
    red = render_report(d, redact=True)
    assert "id0 " not in red and "more (see results.json)" not in red


def test_svg_has_title_and_desc_and_table_headers_have_scope():
    html = render_report(DATA)
    assert "<title>Score versus total cost</title>" in html and "<desc>" in html
    assert 'scope="col"' in html and "<th>" not in html


def test_frontier_svg_marks_highlight_and_escapes_names():
    svg = frontier_svg([{"name": "a<b", "x": 1.0, "y": 0.9, "lo": 0.8, "hi": 0.95, "highlight": True},
                        {"name": "c", "x": 10.0, "y": 0.95, "lo": 0.9, "hi": 0.99, "highlight": False}])
    assert svg.startswith("<svg") and "a&lt;b" in svg and "a<b" not in svg


def test_frontier_svg_handles_zero_cost_single_and_identical_points():
    free = frontier_svg([{"name": "local", "x": 0.0, "y": 0.9, "lo": None, "hi": None, "highlight": True},
                         {"name": "neg", "x": -1.0, "y": 0.8, "lo": 0.7, "hi": 0.85, "highlight": False},
                         {"name": "paid", "x": 5.0, "y": 0.95, "lo": 0.9, "hi": 0.99, "highlight": False}])
    assert "nan" not in free.lower() and "inf" not in free.lower() and "clamp" in free.lower()
    one = frontier_svg([{"name": "solo", "x": 3.0, "y": 0.9, "lo": 0.8, "hi": 0.95, "highlight": False}])
    assert "solo" in one and "nan" not in one.lower() and "clamp" not in one.lower()
    same = frontier_svg([{"name": "p", "x": 2.0, "y": 0.9, "lo": 0.9, "hi": 0.9, "highlight": False},
                         {"name": "q", "x": 2.0, "y": 0.9, "lo": 0.9, "hi": 0.9, "highlight": False}])
    assert "nan" not in same.lower()
    assert frontier_svg([]) == ""
    assert frontier_svg([{"name": "n", "x": 1.0, "y": None, "lo": None, "hi": None, "highlight": False}]) == ""


def test_null_scores_show_na_and_are_left_off_the_chart():
    d = copy.deepcopy(DATA)
    d["candidates"].append({**d["candidates"][1], "name": "dead", "status": "excluded", "note": "80% of calls failed",
                            "score": None, "ci_lo": None, "ci_hi": None, "cost_per_1k": None,
                            "total_cost_at_volume": None, "p50_latency_s": None, "p95_latency_s": None})
    d["warnings"].append("dead: 80% of calls failed; excluded from the comparison")
    html = render_report(d)
    assert "n/a" in html and "80% of calls failed" in html and "excluded" in html
    svg = re.search(r"<svg.*?</svg>", html, re.DOTALL).group(0)
    assert "dead" not in svg and "eliminated" in html


def test_warnings_appear_before_the_candidate_table():
    d = copy.deepcopy(DATA)
    d["incomplete"] = True
    d["warnings"].append("The cost cap was reached; the run is incomplete.")
    html = render_report(d)
    assert html.index("Only 20 examples.") < html.index("<table")
    assert html.index("cost cap was reached") < html.index("<table")
    assert html.index("SIMULATED") < html.index("<table")
    assert html.count("Only 20 examples.") == 1


def test_hostile_names_are_escaped_in_table_svg_and_notes():
    bad = "<script>alert(1)</script>"
    d = copy.deepcopy(DATA)
    d["candidates"][0]["name"] = bad
    d["candidates"][1]["note"] = 'x"><img src=x onerror=1>&'
    d["recommendation"]["winner"] = bad
    d["recommendation"]["equivalent"] = [bad, "big"]
    d["pairwise"][0]["a"] = bad
    d["diagnostics"]["mislabels"][0]["id"] = bad
    d["warnings"].append(bad)
    html = render_report(d)
    assert "<script" not in html.lower() and "<img" not in html.lower()
    assert "&lt;script&gt;" in html
    svg = re.search(r"<svg.*?</svg>", html, re.DOTALL).group(0)
    assert "&lt;script&gt;" in svg


def test_redact_removes_ids_everywhere():
    d = copy.deepcopy(DATA)
    d["diagnostics"]["mislabels"] = [{"id": "secret-id-9", "label": "positive", "disagree_fraction": 1.0}]
    d["diagnostics"]["disagreement_ids"] = ["secret-id-1", "secret-id-2"]
    html = render_report(d, redact=True)
    assert "secret-id" not in html
    assert "secret-id-9" in render_report(d)


def test_write_report_reads_results_json(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(DATA), encoding="utf-8")
    out = write_report(tmp_path)
    assert out.name == "report.html" and out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_report_from_a_real_engine_run(tmp_path):
    df = make_reviews(120, 0)
    ds = Dataset(df["id"].tolist(), df[["review"]], df["sentiment"].to_numpy(dtype=object), df["review"].tolist())
    task = TaskConfig(type="classify", label_column="sentiment", text_column="review",
                      labels=["negative", "neutral", "positive"], repeats=1, cv_folds=3, cv_repeats=1)
    models = [ModelSpec(name="logreg", kind="ml", estimator="logistic_regression"),
              ModelSpec(name="forest", kind="ml", estimator="random_forest"),
              ModelSpec(name="mockbig", kind="llm", provider="mock", model="mockbig",
                        params={"accuracy": 0.95, "cost_per_call": 0.01})]
    res = run_experiment(task, models, ds, tmp_path / "run", sleep=lambda s: None, n_boot=200)
    html = write_report(tmp_path / "run").read_text(encoding="utf-8")
    assert res.data["recommendation"]["winner"] in html
    for name in ("logreg", "forest", "mockbig"):
        assert name in html
    assert "SIMULATED" in html and "<svg" in html and "<script" not in html.lower()
    redacted = render_report(res.data, redact=True)
    ids = res.data["diagnostics"]["disagreement_ids"] + [m["id"] for m in res.data["diagnostics"]["mislabels"]]
    for eid in ids:
        assert not re.search(rf"(?<![\w-]){re.escape(eid)}(?![\w-])", redacted), eid


def test_provisional_recommendation_uses_warning_style_and_heading():
    d = copy.deepcopy(DATA)
    d["recommendation"].update(status="provisional", reason="Provisional: left out of the comparison: big (incomplete: provider errors). x")
    html = render_report(d)
    assert "Provisional choice: logreg" in html and "Recommendation:" not in html
    assert '<div class="card rec">' not in html and "left out of the comparison" in html
