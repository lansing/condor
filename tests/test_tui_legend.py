"""Tests for the legend panel toggle behaviour.

The legend lives inside StackedBarPanel alongside _LatencyBars, occupying
the right 1/3 of the latency panel area.  It is toggled with 'l'.
"""

from __future__ import annotations

import pytest

textual = pytest.importorskip("textual", reason="textual not installed (uv sync --extra tui)")

from condor.tui.app import CondorTUI, LegendPanel  # noqa: E402


@pytest.mark.asyncio
async def test_legend_hidden_by_default():
    async with CondorTUI().run_test(size=(120, 40)) as pilot:
        assert pilot.app.query_one(LegendPanel).display is False


@pytest.mark.asyncio
async def test_l_shows_legend():
    async with CondorTUI().run_test(size=(120, 40)) as pilot:
        await pilot.press("l")
        await pilot.pause()
        assert pilot.app.query_one(LegendPanel).display is True


@pytest.mark.asyncio
async def test_second_l_hides_legend():
    async with CondorTUI().run_test(size=(120, 40)) as pilot:
        await pilot.press("l")
        await pilot.pause()
        await pilot.press("l")
        await pilot.pause()
        assert pilot.app.query_one(LegendPanel).display is False


@pytest.mark.asyncio
async def test_throughput_panel_unaffected_by_legend_toggle():
    """Throughput panel should always remain visible regardless of legend state."""
    from condor.tui.app import GraphPanel
    async with CondorTUI().run_test(size=(120, 40)) as pilot:
        throughput = pilot.app.query_one("#throughput-panel", GraphPanel)
        assert throughput.display is True
        await pilot.press("l")
        await pilot.pause()
        assert throughput.display is True
        await pilot.press("l")
        await pilot.pause()
        assert throughput.display is True


@pytest.mark.asyncio
async def test_legend_is_inside_latency_panel():
    """LegendPanel should be a descendant of StackedBarPanel, not a sibling."""
    from condor.tui.app import StackedBarPanel
    async with CondorTUI().run_test(size=(120, 40)) as pilot:
        latency = pilot.app.query_one(StackedBarPanel)
        # query_one will raise if LegendPanel is not found within latency panel
        latency.query_one(LegendPanel)
