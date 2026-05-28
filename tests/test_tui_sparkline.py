"""Unit tests for StackedBarPanel sparkline rendering.

The bug: _update_ui computed _num_ticks = lat_panel.size.width - 4, but
Widget.size in Textual already returns the content area (border + padding
excluded).  The superfluous -4 meant 4 fewer data points were requested,
leaving 4 blank leading ▁ columns in the rendered output.

Run with:
    uv run pytest tests/test_tui_sparkline.py -v
"""

from __future__ import annotations

import re

import pytest

textual = pytest.importorskip("textual", reason="textual not installed (uv sync --extra tui)")

from textual.app import App, ComposeResult  # noqa: E402

from condor.tui.app import StackedBarPanel, _LatencyBars  # noqa: E402
from condor.tui.palette import load_palette  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal host app — StackedBarPanel fills the entire terminal.
# ---------------------------------------------------------------------------

TERM_W = 60
TERM_H = 12


class _SparkApp(App):
    def __init__(self) -> None:
        self._palette = load_palette()
        super().__init__()

    @property
    def palette(self):
        return self._palette

    def get_css_variables(self) -> dict[str, str]:
        base = super().get_css_variables()
        base.update(self._palette.css_variables())
        return base

    def compose(self) -> ComposeResult:
        yield StackedBarPanel()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MARKUP_RE = re.compile(r"\[.*?\]")


def _strip_markup(s: str) -> str:
    """Remove Rich markup tags to get the plain character content."""
    return _MARKUP_RE.sub("", s)


def _count_leading_blanks(rendered: str) -> int:
    """Return the number of leading ▁ chars in the bottom bar row.

    _LatencyBars.render() emits: title, bar rows (no summary).
    The bottom bar row is the last line.
    """
    lines = rendered.split("\n")
    # lines[0] = title row; lines[1:] = bar rows
    if len(lines) < 2:
        return 0
    bottom_bar = _strip_markup(lines[-1])
    count = 0
    for ch in bottom_bar:
        if ch == "▁":
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_size_is_content_area():
    """Sanity-check: Widget.size in Textual equals the content area (border and
    padding already excluded), so no manual subtraction is needed."""
    async with _SparkApp().run_test(size=(TERM_W, TERM_H)) as pilot:
        panel = pilot.app.query_one(StackedBarPanel)
        # border: heavy (1 each side = 2) + padding: 0 1 (1 each side = 2) = 4 total
        assert panel.size.width == TERM_W - 4, (
            f"Expected content width {TERM_W - 4}, got {panel.size.width}. "
            "Textual's size already excludes border+padding."
        )
        assert panel.size == panel.content_size, (
            "size and content_size should be identical for this widget"
        )


@pytest.mark.asyncio
async def test_undercount_by_four_produces_leading_blanks():
    """Demonstrates the bug: feeding size.width-4 points leaves 4 leading ▁ columns."""
    async with _SparkApp().run_test(size=(TERM_W, TERM_H)) as pilot:
        panel = pilot.app.query_one(StackedBarPanel)
        bars = pilot.app.query_one(_LatencyBars)

        content_w = panel.size.width
        # Simulate what the buggy _update_ui provided: content_w - 4 ticks.
        buggy_n = content_w - 4
        lat_data = [float(i % 5 + 1) for i in range(buggy_n)]
        panel.update_data(lat_data, {}, "")
        await pilot.pause()

        blanks = _count_leading_blanks(bars.render())
        assert blanks == 4, (
            f"Expected 4 leading blanks with buggy tick count, got {blanks}"
        )


@pytest.mark.asyncio
async def test_stacked_bar_no_leading_blanks_in_bordered_container():
    """Fix: feeding size.width points fills the content area with zero leading ▁."""
    async with _SparkApp().run_test(size=(TERM_W, TERM_H)) as pilot:
        panel = pilot.app.query_one(StackedBarPanel)
        bars = pilot.app.query_one(_LatencyBars)

        content_w = panel.size.width
        assert content_w > 0, "content width not available after layout"

        lat_data = [float(i % 5 + 1) for i in range(content_w)]
        panel.update_data(lat_data, {}, "now 3.0  avg 3.0  peak 5.0")
        await pilot.pause()

        blanks = _count_leading_blanks(bars.render())
        assert blanks == 0, (
            f"Bottom bar row has {blanks} leading blank column(s) — "
            f"outer terminal={TERM_W}, content_w={content_w}.\n"
            f"Bottom row (plain): {_strip_markup(bars.render().split(chr(10))[-1])!r}"
        )


@pytest.mark.asyncio
async def test_stacked_bar_no_leading_blanks_with_stage_data():
    """Same check with full per-stage data (stacked mode)."""
    async with _SparkApp().run_test(size=(TERM_W, TERM_H)) as pilot:
        panel = pilot.app.query_one(StackedBarPanel)
        bars = pilot.app.query_one(_LatencyBars)

        content_w = panel.size.width
        lat_data = [5.0] * content_w
        stages = {s: [1.0] * content_w for s in ["mcpy", "h2d", "swait", "exec", "d2h", "pp"]}
        panel.update_data(lat_data, stages, "")
        await pilot.pause()

        rendered = bars.render()
        lines = rendered.split("\n")
        bar_rows = lines[1:]  # skip title; _LatencyBars has no summary line
        assert bar_rows, "Expected at least one bar row"

        for i, row in enumerate(bar_rows):
            plain = _strip_markup(row)
            assert not plain.startswith("▁"), (
                f"Bar row {i} has a leading blank: {plain!r}"
            )
