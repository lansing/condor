from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple


_MARKUP_TAG_RE = re.compile(r"\[/?[^\[\]]+\]")


def _char_cols(ch: str) -> int:
    eaw = unicodedata.east_asian_width(ch)
    return 2 if eaw in ("W", "F") else 1


def _vis(s: str) -> int:
    return sum(_char_cols(ch) for ch in _MARKUP_TAG_RE.sub("", s))


def _trunc(s: str, width: int) -> str:
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
