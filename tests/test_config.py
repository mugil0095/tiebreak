import pytest
from pydantic import ValidationError

from tiebreak.config import ConfigError, ModelSpec, TaskConfig, load_models, load_pricing, load_task


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


def _bad(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.parametrize("loader,name", [(load_task, "task.yaml"), (load_models, "models.yaml")])
def test_empty_file_gives_clear_error(tmp_path, loader, name):
    p = _bad(tmp_path, name, "")
    with pytest.raises(ConfigError, match="empty") as e:
        loader(p)
    assert name in str(e.value)


@pytest.mark.parametrize("loader,name", [(load_task, "task.yaml"), (load_models, "models.yaml"),
                                         (load_pricing, "pricing.yaml")])
def test_malformed_yaml_names_file(tmp_path, loader, name):
    p = _bad(tmp_path, name, "a: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML") as e:
        loader(p)
    assert name in str(e.value)


def test_missing_required_key_names_file_and_field(tmp_path):
    p = _bad(tmp_path, "task.yaml", "type: classify\nlabels: [a, b]\n")
    with pytest.raises(ConfigError) as e:
        load_task(p)
    assert "task.yaml" in str(e.value) and "label_column" in str(e.value)
    assert not isinstance(e.value, ValidationError)


def test_models_wrong_shape_and_bad_entry(tmp_path):
    p = _bad(tmp_path, "models.yaml", "name: a\nkind: ml\n")
    with pytest.raises(ConfigError, match="list"):
        load_models(p)
    p = _bad(tmp_path, "models2.yaml", "- {name: a, kind: llm}\n")
    with pytest.raises(ConfigError, match="models2.yaml"):
        load_models(p)
    p = _bad(tmp_path, "models3.yaml", "- just a string\n")
    with pytest.raises(ConfigError, match="models3.yaml"):
        load_models(p)


def test_malformed_pricing_entry(tmp_path):
    p = _bad(tmp_path, "pricing.yaml", "m: {input_per_1k: 0.5}\n")
    with pytest.raises(ConfigError, match="output_per_1k") as e:
        load_pricing(p)
    assert "pricing.yaml" in str(e.value)
    p = _bad(tmp_path, "pricing2.yaml", "m: 3\n")
    with pytest.raises(ConfigError, match="pricing2.yaml"):
        load_pricing(p)
    p = _bad(tmp_path, "pricing3.yaml", "m: {input_per_1k: abc, output_per_1k: 1}\n")
    with pytest.raises(ConfigError, match="pricing3.yaml"):
        load_pricing(p)


def test_missing_file_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_task(tmp_path / "nope.yaml")


def test_requests_per_minute_must_be_positive():
    ok = ModelSpec(name="m", kind="llm", provider="p", model="x", requests_per_minute=10)
    assert ok.requests_per_minute == 10
    with pytest.raises(ValidationError):
        ModelSpec(name="m", kind="llm", provider="p", model="x", requests_per_minute=0)
