import numpy as np
import pytest

from tiebreak.config import TaskConfig
from tiebreak.data import DataError, concat, load_dataset

TASK = TaskConfig(type="classify", label_column="s", text_column="t", labels=["a", "b"])


def write(tmp_path, text, name="d.csv"):
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8"))
    return p


def test_loads_csv_with_bom_and_generates_ids(tmp_path):
    p = write(tmp_path, "﻿t,s\nhello,a\nworld,b\n")
    ds = load_dataset(p, TASK)
    assert ds.ids == ["row-0", "row-1"] and list(ds.y) == ["a", "b"]
    assert ds.text == ["hello", "world"]


def test_uses_id_column_when_unique(tmp_path):
    ds = load_dataset(write(tmp_path, "id,t,s\nx1,hi,a\nx2,yo,b\n"), TASK)
    assert ds.ids == ["x1", "x2"] and "id" not in ds.frame.columns


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(DataError, match="duplicate"):
        load_dataset(write(tmp_path, "id,t,s\nx,hi,a\nx,yo,b\n"), TASK)


def test_missing_labels_name_the_rows(tmp_path):
    with pytest.raises(DataError, match=r"no label.*\[2\]"):
        load_dataset(write(tmp_path, "t,s\nhi,a\nyo,\nok,b\n"), TASK)


def test_labels_outside_task_labels_rejected(tmp_path):
    with pytest.raises(DataError, match="not in task.labels"):
        load_dataset(write(tmp_path, "t,s\nhi,a\nyo,zzz\n"), TASK)


def test_single_class_rejected(tmp_path):
    with pytest.raises(DataError, match="at least two"):
        load_dataset(write(tmp_path, "t,s\nhi,a\nyo,a\n"), TASK)


def test_empty_text_rejected(tmp_path):
    with pytest.raises(DataError, match="empty text"):
        load_dataset(write(tmp_path, "t,s\nhi,a\n,b\n"), TASK)


def test_missing_label_column(tmp_path):
    with pytest.raises(DataError, match="label column"):
        load_dataset(write(tmp_path, "t,x\nhi,a\n"), TASK)


def test_jsonl_and_regression(tmp_path):
    task = TaskConfig(type="regress", label_column="v")
    p = write(tmp_path, '{"x": 1, "v": 2.5}\n{"x": 2, "v": 3.5}\n', "d.jsonl")
    ds = load_dataset(p, task)
    assert ds.y.dtype == float and list(ds.y) == [2.5, 3.5]


def test_take_and_concat(tmp_path):
    ds = load_dataset(write(tmp_path, "t,s\nhi,a\nyo,b\nok,a\n"), TASK)
    sub = ds.take([2, 0])
    assert sub.ids == ["row-2", "row-0"] and list(sub.y) == ["a", "a"]
    both = concat(sub, ds.take([1]))
    assert len(both) == 3 and both.text == ["ok", "hi", "yo"]
    assert concat(sub, None) is sub


def test_label_none_is_a_real_label_not_missing(tmp_path):
    task = TaskConfig(type="classify", label_column="s", text_column="t", labels=["None", "b"])
    ds = load_dataset(write(tmp_path, "t,s\nhi,None\nyo,b\n"), task)
    assert list(ds.y) == ["None", "b"]


def test_text_na_is_accepted(tmp_path):
    ds = load_dataset(write(tmp_path, "t,s\nN/A,a\nnull,b\n"), TASK)
    assert ds.text == ["N/A", "null"]


def test_id_leading_zeros_preserved_and_distinct_from_seven(tmp_path):
    ds = load_dataset(write(tmp_path, "id,t,s\n007,hi,a\n7,yo,b\n"), TASK)
    assert ds.ids == ["007", "7"]


def test_blank_id_rejected(tmp_path):
    with pytest.raises(DataError, match=r"blank id.*\[1\]"):
        load_dataset(write(tmp_path, "id,t,s\n,hi,a\nx2,yo,b\n"), TASK)


def test_duplicate_id_message_names_the_id(tmp_path):
    with pytest.raises(DataError, match=r"duplicate.*'x'"):
        load_dataset(write(tmp_path, "id,t,s\nx,hi,a\nx,yo,b\n"), TASK)


def test_empty_file_gives_data_error(tmp_path):
    with pytest.raises(DataError, match="could not read d.csv"):
        load_dataset(write(tmp_path, ""), TASK)


def test_garbled_jsonl_gives_data_error(tmp_path):
    task = TaskConfig(type="regress", label_column="v")
    with pytest.raises(DataError, match="could not read d.jsonl"):
        load_dataset(write(tmp_path, "not json at all\n", "d.jsonl"), task)


def test_inf_regression_label_rejected(tmp_path):
    task = TaskConfig(type="regress", label_column="v")
    with pytest.raises(DataError, match="non-finite"):
        load_dataset(write(tmp_path, "t,v\nhi,inf\nyo,2\n"), task)
