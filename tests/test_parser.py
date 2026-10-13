import pytest

from tiebreak.config import TaskConfig
from tiebreak.parser import parse_answer, parse_label, parse_number, parse_text

LABELS = ["negative", "neutral", "positive"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Positive.", "positive"),
        ("**neutral**", "neutral"),
        ("positive\nbecause it is great", "positive"),
        ("The sentiment is negative.", "negative"),
        ('{"label": "Neutral"}', "neutral"),
        ('{"answer": "positive"}', "positive"),
    ],
)
def test_parse_label_valid(text, expected):
    assert parse_label(text, LABELS) == expected


@pytest.mark.parametrize("text", ["", "   ", "I am not sure", "positive or negative", "unpositive"])
def test_parse_label_invalid(text):
    assert parse_label(text, LABELS) is None


def test_parse_number():
    assert parse_number("The answer is 3.5.") == 3.5
    assert parse_number("-2") == -2.0
    assert parse_number("1,234.5 units") == 1234.5
    assert parse_number("none") is None


def test_parse_text():
    assert parse_text('"Paris"\nextra line') == "Paris"
    assert parse_text("  ") is None


def test_parse_answer_dispatch():
    cls = TaskConfig(type="classify", label_column="y", labels=LABELS)
    assert parse_answer("Positive", cls) == "positive"
    assert parse_answer("7", TaskConfig(type="regress", label_column="y")) == 7.0
    assert parse_answer("Rome", TaskConfig(type="extract_exact", label_column="y")) == "Rome"
