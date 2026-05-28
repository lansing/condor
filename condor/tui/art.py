from __future__ import annotations

import random
import re
import unicodedata
from typing import NamedTuple


_MARKUP_TAG_RE = re.compile(r"\[/?[^\[\]]+\]")


def _char_cols(ch: str) -> int:
    """Return the terminal column width of a single character (1 or 2)."""
    eaw = unicodedata.east_asian_width(ch)
    return 2 if eaw in ("W", "F") else 1


def _vis(s: str) -> int:
    """Terminal column count of a Rich markup string (markup tags excluded)."""
    return sum(_char_cols(ch) for ch in _MARKUP_TAG_RE.sub("", s))


def _trunc(s: str, width: int) -> str:
    """Truncate a Rich markup string to *width* terminal columns.

    If the string already fits, it is returned unchanged (markup intact).
    If truncation is needed, markup is stripped and characters are consumed
    until the budget is exhausted, then a trailing ellipsis is appended.
    """
    if width <= 0:
        return ""
    plain = _MARKUP_TAG_RE.sub("", s)
    if _vis(s) <= width:
        return s
    # Need to truncate — build plain result character by character
    result: list[str] = []
    cols = 0
    for ch in plain:
        w = _char_cols(ch)
        if cols + w > width - 1:  # reserve 1 col for ellipsis
            break
        result.append(ch)
        cols += w
    return "".join(result) + "…"


def build_combined_logo(condor_lines: list[str], status_lines: list[str], total_width: int = 90) -> str:
    """Combine CONDOR logo (left) and status lines (right) into a single string.

    Lines that would overflow the available width are truncated with an ellipsis
    so the output never wraps inside the header widget.
    """
    condor_width = 54  # CONDOR logo is ~51 chars wide; leaves a small gap
    status_width = max(total_width - condor_width, 0)

    combined_lines = []
    for i in range(max(len(condor_lines), len(status_lines))):
        condor_line = condor_lines[i] if i < len(condor_lines) else ""
        status_line = status_lines[i] if i < len(status_lines) else ""

        condor_padded = condor_line.ljust(condor_width)
        status_fitted = _trunc(status_line, status_width)
        combined_lines.append(f"{condor_padded}{status_fitted}")

    return "\n".join(combined_lines)
