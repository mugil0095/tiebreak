import re

_STRIP = " \t\r\n*_`\"'“”‘’.,;:!?()[]{}"


def normalize(s) -> str:
    s = re.sub(r"\s+", " ", str(s).casefold()).strip()
    return s.strip(_STRIP)
