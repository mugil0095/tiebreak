REPO_URL = "git+https://github.com/mugil0095/tiebreak.git"


def install_hint(extra: str) -> str:
    """The command that installs an optional extra. tiebreak is not on PyPI, so it installs from GitHub."""
    return f'pip install "tiebreak[{extra}] @ {REPO_URL}"'
