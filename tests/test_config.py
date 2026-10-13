import pytest
from pydantic import ValidationError

from tiebreak.config import ModelSpec, TaskConfig, load_models, load_pricing, load_task


def test_task_defaults():
    t = TaskConfig(type="classify", label_column="y", labels=["a", "b", "c"])
    assert t.metric_name == "macro_f1"
    assert t.margin == 0.02 and t.repeats == 3 and t.volume == 10_000


def test_default_metrics_by_type():
    assert TaskConfig(type="binary", label_column="y", labels=["a", "b"]).metric_name == "accuracy"
    assert TaskConfig(type="regress", label_column="y").metric_name == "neg_rmse"
    assert TaskConfig(type="extract_exact", label_column="y").metric_name == "accuracy"


def test_classify_needs_labels():
    with pytest.raises(ValidationError):
        TaskConfig(type="classify", label_column="y")


def test_binary_needs_exactly_two_labels():
    with pytest.raises(ValidationError):
        TaskConfig(type="binary", label_column="y", labels=["a", "b", "c"])


def test_llm_needs_provider_and_model():
    with pytest.raises(ValidationError):
        ModelSpec(name="m", kind="llm")


def test_ml_needs_estimator():
    with pytest.raises(ValidationError):
        ModelSpec(name="m", kind="ml")


def test_load_models_rejects_duplicate_names(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(
        "- {name: a, kind: ml, estimator: logistic_regression}\n"
        "- {name: a, kind: ml, estimator: random_forest}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_models(p)


def test_load_task_and_pricing(tmp_path):
    (tmp_path / "task.yaml").write_text(
        "type: classify\nlabel_column: s\ntext_column: t\nlabels: [a, b]\nmargin: 0.05\n"
    )
    (tmp_path / "pricing.yaml").write_text(
        "some-model: {input_per_1k: 0.5, output_per_1k: 1.5}\n"
    )
    t = load_task(tmp_path / "task.yaml")
    assert t.margin == 0.05 and t.text_column == "t"
    assert load_pricing(tmp_path / "pricing.yaml") == {"some-model": (0.5, 1.5)}
    assert load_pricing(None) == {}
