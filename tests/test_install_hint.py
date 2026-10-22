import builtins
import re

import pytest
from typer.testing import CliRunner

from tiebreak.cli import app
from tiebreak.install import REPO_URL, install_hint

runner = CliRunner()


def _flat(text: str) -> str:
    return "".join(text.split())  # terminal wrapping may insert line breaks


def test_install_hint_installs_an_extra_from_github_not_pypi():
    assert install_hint("llm") == 'pip install "tiebreak[llm] @ git+https://github.com/mugil0095/tiebreak.git"'
    assert REPO_URL == "git+https://github.com/mugil0095/tiebreak.git"


def test_the_web_hint_names_the_github_install(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] in {"uvicorn", "fastapi"}:
            raise ImportError("no module")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    r = runner.invoke(app, ["serve"])
    assert r.exit_code == 2 and "tiebreak[web]" in _flat(r.output) and REPO_URL in _flat(r.output)


def test_the_missing_litellm_hint_names_the_github_install(monkeypatch):
    from tiebreak.candidates.llm import LiteLLMCompleter

    real = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] == "litellm":
            raise ImportError("no module")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=re.escape(REPO_URL)):
        LiteLLMCompleter("ollama", "m")


def test_the_models_listing_shows_a_real_boost_hint_not_a_template():
    r = runner.invoke(app, ["models"])
    assert "{install_hint" not in r.output and REPO_URL in _flat(r.output)
