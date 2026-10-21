import numpy as np
import pytest

from tiebreak.config import ConfigError, ModelSpec, TaskConfig, load_task, validate_models_for_task
from tiebreak.data import DataError, load_dataset

JUDGE_TASK = {"type": "judge", "prompt_column": "prompt"}


def _llm(name, role="candidate", provider="mock", model="m", **kw):
    return ModelSpec(name=name, kind="llm", provider=provider, model=model, role=role, **kw)


def test_judge_task_minimal_yaml_loads(tmp_path):
    p = tmp_path / "task.yaml"
    p.write_text("type: judge\nprompt_column: prompt\n", encoding="utf-8")
    t = load_task(p)
    assert t.type == "judge" and t.label_column is None
    assert t.margin == 0.25 and t.scale == (1, 5) and t.judge_confidence_min == 0.5


def test_explicit_margin_is_kept_for_judge_tasks():
    assert TaskConfig(**JUDGE_TASK, margin=0.1).margin == 0.1


def test_label_tasks_keep_their_defaults_and_still_need_a_label_column(tmp_path):
    assert TaskConfig(type="classify", label_column="y", labels=["a", "b"]).margin == 0.02
    p = tmp_path / "t.yaml"
    p.write_text("type: classify\nlabels: [a, b]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="label_column"):
        load_task(p)


def test_judge_needs_prompt_column():
    with pytest.raises(ValueError, match="prompt_column"):
        TaskConfig(type="judge")


def test_scale_must_be_increasing():
    with pytest.raises(ValueError, match="scale"):
        TaskConfig(**JUDGE_TASK, scale=(5, 1))
    with pytest.raises(ValueError, match="scale"):
        TaskConfig(**JUDGE_TASK, scale=(3, 3))


def test_exactly_one_judge_model():
    task = TaskConfig(**JUDGE_TASK)
    validate_models_for_task(task, [_llm("a"), _llm("j", role="judge")])
    with pytest.raises(ConfigError, match="judge"):
        validate_models_for_task(task, [_llm("a"), _llm("b")])
    with pytest.raises(ConfigError, match="judge"):
        validate_models_for_task(task, [_llm("a"), _llm("j1", role="judge"), _llm("j2", role="judge")])


def test_judge_task_needs_a_candidate():
    with pytest.raises(ConfigError, match="candidate"):
        validate_models_for_task(TaskConfig(**JUDGE_TASK), [_llm("j", role="judge")])


def test_ml_candidate_rejected_for_judge_task():
    ml = ModelSpec(name="lr", kind="ml", estimator="logistic_regression")
    with pytest.raises(ConfigError, match="ml"):
        validate_models_for_task(TaskConfig(**JUDGE_TASK), [ml, _llm("a"), _llm("j", role="judge")])


def test_judge_role_rejected_on_label_task():
    task = TaskConfig(type="binary", label_column="y", labels=["a", "b"])
    with pytest.raises(ConfigError, match="judge"):
        validate_models_for_task(task, [_llm("a"), _llm("j", role="judge")])


def test_label_tasks_without_a_judge_pass_validation():
    task = TaskConfig(type="binary", label_column="y", labels=["a", "b"])
    validate_models_for_task(task, [_llm("a"), ModelSpec(name="lr", kind="ml", estimator="logistic_regression")])


def test_judge_role_must_be_an_llm():
    with pytest.raises(ValueError, match="judge"):
        ModelSpec(name="j", kind="ml", estimator="ridge", role="judge")


def test_judge_dataset_loads_prompts_without_labels(tmp_path):
    p = tmp_path / "prompts.csv"
    p.write_text("id,prompt,human\nq1,Say hi,4\nq2,Say bye,\n", encoding="utf-8")
    ds = load_dataset(p, TaskConfig(**JUDGE_TASK, human_score_column="human"))
    assert ds.ids == ["q1", "q2"] and ds.text == ["Say hi", "Say bye"]
    assert np.all(ds.y == 5.0)  # the synthetic target: the top of the scale
    assert list(ds.frame["human"].fillna("")) == ["4", ""]


def test_judge_dataset_rejects_missing_prompt_column_and_empty_prompts(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("question\nhi\n", encoding="utf-8")
    with pytest.raises(DataError, match="prompt column 'prompt'"):
        load_dataset(p, TaskConfig(**JUDGE_TASK))
    p.write_text("id,prompt\nq1,hi\nq2,\n", encoding="utf-8")
    with pytest.raises(DataError, match="empty"):
        load_dataset(p, TaskConfig(**JUDGE_TASK))


def test_judge_dataset_rejects_a_missing_human_score_column(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("id,prompt\nq1,hi\nq2,bye\n", encoding="utf-8")
    with pytest.raises(DataError, match="human score column 'human'"):
        load_dataset(p, TaskConfig(**JUDGE_TASK, human_score_column="human"))
