import numpy as np

from tiebreak.diagnostics import class_balance, disagreements, sample_warnings, suspected_mislabels


def test_suspected_mislabels_need_near_unanimous_disagreement():
    y = np.array([0, 1, 0, 1])
    preds = {"a": np.array([0, 1, 1, 1]), "b": np.array([0, 1, 1, 1]), "c": np.array([0, 1, 1, 0])}
    out = suspected_mislabels(["i0", "i1", "i2", "i3"], y, preds)
    assert [o["id"] for o in out] == ["i2"] and out[0]["disagree_fraction"] == 1.0


def test_single_candidate_cannot_flag_mislabels():
    assert suspected_mislabels(["a"], np.array([0]), {"only": np.array([1])}) == []


def test_invalid_predictions_do_not_count_as_disagreement_votes():
    y = np.array([0, 0])
    preds = {"a": np.array([-1, 0]), "b": np.array([-1, 0])}
    assert suspected_mislabels(["a", "b"], y, preds) == []


def test_disagreements():
    preds = {"a": np.array([0, 1, 1]), "b": np.array([0, 0, 1])}
    assert disagreements(["x", "y", "z"], preds) == ["y"]


def test_class_balance():
    assert class_balance(np.array(["a", "a", "b"], dtype=object)) == {"a": 2, "b": 1}


def test_sample_warnings():
    w = sample_warnings(30, 0.02, {"a": 28, "b": 2}, {"m": 0.2})
    text = " ".join(w)
    assert "fewer than 50" in text and "imbalanced" in text and "wider than the margin" in text
    assert "resemble" in text
    assert sample_warnings(500, 0.02, {"a": 250, "b": 250}, {"m": 0.01}) == [
        "Make sure this sample resembles the data you will see in production; results only hold for similar data."
    ]


def test_two_votes_are_enough_to_flag():
    y = np.array([0])
    assert [o["id"] for o in suspected_mislabels(["a"], y, {"a": np.array([1]), "b": np.array([1])})] == ["a"]


def test_invalid_vote_mixed_with_valid_votes():
    y = np.array([0])
    preds = {"a": np.array([-1]), "b": np.array([1]), "c": np.array([1])}
    assert suspected_mislabels(["a"], y, preds)[0]["disagree_fraction"] == 1.0
    assert suspected_mislabels(["a"], y, {"a": np.array([-1]), "b": np.array([1])}) == []


def test_disagreements_empty_and_invalid_is_distinct():
    assert disagreements(["x"], {}) == []
    assert disagreements(["x"], {"a": np.array([-1]), "b": np.array([0])}) == ["x"]


def test_sample_warnings_zero_count_class():
    assert any("imbalanced" in w for w in sample_warnings(500, 0.02, {"a": 500, "b": 0}, {}))


def test_imbalance_warning_does_not_mention_a_macro_f1_column_and_width_compares_to_the_margin():
    w = sample_warnings(500, 0.02, {"a": 450, "b": 10}, {"m": 0.03})
    text = " ".join(w)
    assert "plain accuracy can mislead on imbalanced classes; consider macro_f1 as the metric" in text
    assert "column" not in text
    assert "wider than the margin for: m" in text  # 0.03 > margin 0.02, though not > 2 x margin
    assert "wider than" not in " ".join(sample_warnings(500, 0.02, {"a": 250, "b": 250}, {"m": 0.02}))
