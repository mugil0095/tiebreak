import pytest
from typer.testing import CliRunner

from tiebreak import prices
from tiebreak.cli import app

runner = CliRunner()


@pytest.fixture
def fake_table(monkeypatch):
    table = {
        "gpt-x": {"input_cost_per_token": 1.5e-07, "output_cost_per_token": 6e-07},
        "gemini/gem-y": {"input_cost_per_token": 3e-07, "output_cost_per_token": 2.5e-06},
        "no-prices": {"max_tokens": 10},
    }
    monkeypatch.setattr(prices, "_litellm_table", lambda: table)
    return table


def test_jev_price_is_builtin(fake_table):
    row = prices.lookup_price("typesafe", "jev-1.14.0")
    assert row.input_per_mtok == 0.042 and row.output_per_mtok == 0.0
    assert row.source == "typesafe docs"
    assert prices.lookup_price(None, "jev-1.13.0").input_per_mtok == 0.042


def test_litellm_entry_is_converted_to_per_million(fake_table):
    row = prices.lookup_price(None, "gpt-x")
    assert row.input_per_mtok == pytest.approx(0.15) and row.output_per_mtok == pytest.approx(0.6)
    assert row.source == "litellm table"


def test_provider_prefixed_key_is_tried_first(fake_table):
    assert prices.lookup_price("gemini", "gem-y").input_per_mtok == pytest.approx(0.3)


def test_unknown_and_priceless_models_are_none(fake_table):
    assert prices.lookup_price(None, "nope") is None
    assert prices.lookup_price(None, "no-prices") is None


def test_search_is_case_insensitive_and_limited(fake_table):
    names = [r.model for r in prices.search_prices("GEM")]
    assert names == ["gemini/gem-y"]
    assert len(prices.search_prices("", limit=2)) == 2


def test_without_litellm_only_builtin_entries_exist(monkeypatch):
    monkeypatch.setattr(prices, "_litellm_table", lambda: {})
    assert prices.lookup_price(None, "gpt-x") is None
    assert prices.lookup_price("typesafe", "jev-1.14.0") is not None


def test_models_command_prints_table(fake_table):
    r = runner.invoke(app, ["models", "jev"])
    assert r.exit_code == 0
    assert "jev-1.14.0" in r.output and "jev-1.13.0" in r.output and "typesafe docs" in r.output


def test_models_command_says_when_nothing_matches(fake_table):
    r = runner.invoke(app, ["models", "zzzz"])
    assert r.exit_code == 0 and "no models match" in r.output.lower()
