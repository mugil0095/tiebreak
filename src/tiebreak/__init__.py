__version__ = "0.3.0"


def __getattr__(name):  # lazy: importing tiebreak must stay light
    if name == "compare":
        from .notebook import compare
        return compare
    raise AttributeError(f"module 'tiebreak' has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), "compare"])
