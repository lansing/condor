"""Condor metrics TUI — 90s BBS ANSI-art style.

Connects to the stats Unix socket at /tmp/condor-metrics.sock and displays
live metrics from the running Condor server.

Usage:
    condor-tui
    uv run condor-tui

Requires:
    uv sync --extra tui
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Sparkline, Static

from ..stats import SOCKET_PATH as _DEFAULT_SOCKET_PATH
from .art import _trunc, _vis

# Allow override via env var so the host TUI can reach a socket that is
# bind-mounted from a running Docker container (see docker-compose.yaml).
SOCKET_PATH = os.environ.get("CONDOR_STATS_SOCKET", _DEFAULT_SOCKET_PATH)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Stacked-bar sparkline — stage colours and helpers
# ---------------------------------------------------------------------------

# Easy to reconfigure: change a colour here and it applies everywhere.
STAGE_COLORS: dict[str, str] = {
    "mcpy": "yellow",
    "h2d": "cyan",
    "swait": "red",
    "exec": "blue",
    "d2h": "magenta",
    "pp": "green",
}
# Pipeline execution order — determines top-to-bottom stack order in the bar.
STAGE_ORDER: list[str] = ["mcpy", "h2d", "swait", "exec", "d2h", "pp"]
STAGE_LABELS: dict[str, str] = {
    "mcpy": "Host memory copy",
    "h2d": "Host → Device (H2D)",
    "swait": "GPU queue wait",
    "exec": "TRT engine execute",
    "d2h": "Device → Host (D2H)",
    "pp": "Post-process",
}
STAGE_ABBREV: dict[str, str] = {
    "mcpy": "MCpy",
    "h2d": "H2D",
    "swait": "SWait",
    "exec": "Exec",
    "d2h": "D2H",
    "pp": "PostP",
}
_BLOCK = "█"
_BASELINE_CHAR = "▁"
_BASELINE_COLOR = "_baseline"  # sentinel — not a real Rich colour


def _alloc_rows(vals: dict[str, float], bar_h: int) -> dict[str, int]:
    """Allocate *bar_h* rows to stages proportionally (descending-first greedy)."""
    D = sum(vals.values())
    if D == 0 or bar_h == 0:
        return {s: 0 for s in vals}

    sorted_stages = sorted(vals.items(), key=lambda x: x[1], reverse=True)
    rows: dict[str, int] = {s: 0 for s in vals}
    used = 0
    for stage, v in sorted_stages:
        r = round(v / D * bar_h)
        r = min(r, bar_h - used)
        rows[stage] = r
        used += r
        if used >= bar_h:
            break
    # Rounding shortfall → give remainder to largest stage
    if used < bar_h and sorted_stages:
        rows[sorted_stages[0][0]] += bar_h - used
    return rows


def _build_column(
    vals: dict[str, float], bar_h: int, e2e: float, peak: float
) -> list[str]:
    """Return a list of *bar_h* colour strings (or '' for empty) for one bar column.

    Row 0 = top, row bar_h-1 = bottom.  Bar height is scaled by e2e/peak
    (bottom-aligned).  Within the bar, segments are proportional to stage shares.
    """
    col: list[str] = [""] * bar_h
    if bar_h == 0:
        return col

    # No data → baseline marker only
    if e2e <= 0 or peak <= 0:
        col[bar_h - 1] = _BASELINE_COLOR
        return col

    # Scale total bar height relative to peak (bottom-aligned)
    bar_total = max(1, min(bar_h, round(e2e / peak * bar_h)))
    start_row = bar_h - bar_total

    D = sum(vals.values())
    if D == 0:
        # --- Fallback / future per-provider hook ---
        # No stage data (non-TRT): single-colour E2E bar, height already scaled.
        for r in range(start_row, bar_h):
            col[r] = STAGE_COLORS["exec"]
        return col

    # Full stacked mode: allocate bar_total rows proportionally, bottom-aligned.
    alloc = _alloc_rows(vals, bar_total)
    cur_row = start_row
    for stage in STAGE_ORDER:
        n = alloc.get(stage, 0)
        if n > 0:
            color = STAGE_COLORS[stage]
            for r in range(cur_row, min(cur_row + n, bar_h)):
                col[r] = color
            cur_row += n
    return col


def _render_bar_row(row: list[str]) -> str:
    """Convert a list of colour strings to a Rich-markup line of block characters."""
    if not row:
        return ""
    parts: list[str] = []
    i = 0
    while i < len(row):
        color = row[i]
        j = i + 1
        while j < len(row) and row[j] == color:
            j += 1
        span = j - i
        if color == _BASELINE_COLOR:
            parts.append(f"[dim]{_BASELINE_CHAR * span}[/dim]")
        elif color:
            parts.append(f"[{color}]{_BLOCK * span}[/{color}]")
        else:
            parts.append(" " * span)
        i = j
    return "".join(parts)


def _fmt_ms_row(d: dict) -> str:
    """Format an avg_p99 dict as a fixed-width string."""
    return f"{d['avg']:6.1f}  {d['p99']:6.1f}"


# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------


class StatusBanner(Static):
    """Single-line status bar: connection state, uptime, workers, model."""

    DEFAULT_CSS = """
    StatusBanner {
        height: 3;
        border: heavy $primary;
        color: $primary;
        content-align: left middle;
        background: $background;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._uptime = 0.0
        self._workers_active = 0
        self._num_workers = 0
        self._model = ""
        self._state = "connecting"

    def update_status(
        self, uptime: float, workers_active: int, num_workers: int, model: str
    ) -> None:
        self._uptime = uptime
        self._workers_active = workers_active
        self._num_workers = num_workers
        self._model = model
        self._state = "online"
        self.refresh()

    def update_disconnected(self) -> None:
        self._state = "disconnected"
        self.refresh()

    def render(self) -> str:
        if self._state == "connecting":
            return "[dim]● CONNECTING…[/dim]"
        if self._state == "disconnected":
            return (
                "[bold red]● DISCONNECTED[/bold red]  "
                f"[dim]waiting for condor server at {SOCKET_PATH}…[/dim]"
            )
        prefix = (
            f"[bold green]● ONLINE[/bold green]  "
            f"⏱ [cyan]{_fmt_time(self._uptime)}[/cyan]  "
            f"⚙ [yellow]{self._workers_active}/{self._num_workers} workers[/yellow]  "
            f"📦 "
        )
        model_budget = self.size.width - _vis(prefix)
        model = _trunc(self._model, model_budget)
        return f"{prefix}[white]{model}[/white]"


# ---------------------------------------------------------------------------
# Stacked-bar latency panel
# ---------------------------------------------------------------------------


class _LatencyBars(Static):
    """Renders the title row and stacked bar chart (no summary line)."""

    DEFAULT_CSS = """
    _LatencyBars { width: 2fr; height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._lat_data: list[float] = []
        self._stages: dict[str, list[float]] = {}

    def update(self, lat_data: list[float], stages: dict[str, list[float]]) -> None:
        self._lat_data = lat_data
        self._stages = stages
        self.refresh()

    def render(self) -> str:  # type: ignore[override]
        lat = self._lat_data
        stages = self._stages

        # Widget.size is the content area (Textual excludes border and padding).
        # bar_h: content_h - title(1)  [summary lives in a sibling Static]
        # bar_w: content_w — matches _num_ticks = lat_panel.size.width in _update_ui
        bar_h = max(1, self.size.height - 1)
        bar_w = max(1, self.size.width)

        peak = max(lat) if lat else 0.0
        title = f" ▶ E2E LATENCY  [yellow]↑ {peak:.0f}[/yellow]"

        if not lat:
            return title

        # Align to rightmost n_cols ticks; left-pad to bar_w with baseline markers
        n_cols = min(bar_w, len(lat))
        offset = len(lat) - n_cols
        lat_slice = lat[offset:]
        n_blank = bar_w - n_cols

        # Grid is always bar_w wide so rendered rows fill the content area exactly
        grid: list[list[str]] = [[""] * bar_w for _ in range(bar_h)]
        for col in range(n_blank):
            grid[bar_h - 1][col] = _BASELINE_COLOR
        for col_idx in range(n_cols):
            col = n_blank + col_idx
            t_idx = offset + col_idx
            vals: dict[str, float] = {
                stage: (
                    stages.get(stage, [])[t_idx]
                    if t_idx < len(stages.get(stage, []))
                    else 0.0
                )
                for stage in STAGE_ORDER
            }
            col_colors = _build_column(vals, bar_h, lat_slice[col_idx], peak)
            for row in range(bar_h):
                grid[row][col] = col_colors[row]

        lines = [title]
        for row in grid:
            lines.append(_render_bar_row(row))
        return "\n".join(lines)


class StackedBarPanel(Widget):
    """E2E latency sparkline rendered as a stacked pipeline-stage bar chart.

    Each vertical bar represents one tick.  Its height segments show the
    relative share of each pipeline stage (MCpy → H2D → SWait → Exec → D2H →
    PostP) for that tick, using the colours in STAGE_COLORS.

    When no stage data is available (non-TRT providers), falls back to a
    single-colour E2E bar scaled to the peak value.
    """

    DEFAULT_CSS = """
    StackedBarPanel {
        width: 1fr;
        height: 1fr;
        border: heavy $success;
        padding: 0 1;
        background: $background;
        color: $success;
    }
    StackedBarPanel > Horizontal {
        height: 1fr;
    }
    StackedBarPanel > .summary {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="latency-panel")
        self._n: int = 60

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield _LatencyBars()
            yield LegendPanel()
        yield Static("", classes="summary")

    def update_data(
        self,
        lat_data: list[float],
        stages: dict[str, list[float]],
        n: int,
        summary: str,
    ) -> None:
        self._n = n
        self.query_one(_LatencyBars).update(lat_data, stages)
        self.query_one(".summary", Static).update(summary)


# ---------------------------------------------------------------------------
# Sparkline panels
# ---------------------------------------------------------------------------


class GraphPanel(Widget):
    """Labeled sparkline panel with a live max y-scale label."""

    DEFAULT_CSS = """
    GraphPanel {
        height: 1fr;
        border: heavy $accent;
        padding: 0 1;
        background: $background;
    }
    GraphPanel > .title {
        height: 1;
        color: $accent;
        text-style: bold;
    }
    GraphPanel > Sparkline {
        height: 1fr;
    }
    GraphPanel > .summary {
        height: 1;
        color: $text-muted;
    }
    #throughput-panel-spark > .sparkline--max-color {
        color: $accent;
    }
    #throughput-panel-spark > .sparkline--min-color {
        color: $accent 30%;
    }
    """

    def __init__(self, title: str, unit: str, widget_id: str) -> None:
        super().__init__(id=widget_id)
        self._title = title
        self._unit = unit
        self._title_id = f"{widget_id}-title"
        self._spark_id = f"{widget_id}-spark"
        self._summary_id = f"{widget_id}-summary"

    def compose(self) -> ComposeResult:
        yield Static(f" ▶ {self._title}", id=self._title_id, classes="title")
        yield Sparkline([], id=self._spark_id, summary_function=max)
        yield Static("", id=self._summary_id, classes="summary")

    def update_data(self, data: list[float], summary: str) -> None:
        if not data:
            return
        # Show the current max value as a y-axis scale hint in the title
        peak = max(data)
        scale = f"{peak:.0f}"
        self.query_one(f"#{self._title_id}", Static).update(
            f" ▶ {self._title}  [dim]↑ {scale}[/dim]"
        )
        self.query_one(f"#{self._spark_id}", Sparkline).data = data
        self.query_one(f"#{self._summary_id}", Static).update(summary)


# ---------------------------------------------------------------------------
# Per-worker panel
# ---------------------------------------------------------------------------


class WorkerPanel(Static):
    """Displays stats for one worker thread."""

    DEFAULT_CSS = """
    WorkerPanel {
        width: 1fr;
        height: 100%;
        border: double $success;
        padding: 0 1;
        background: $background;
        color: $text;
    }
    """

    def __init__(self, worker_id: int, port: int) -> None:
        super().__init__(id=f"worker-panel-{worker_id}")
        self._worker_id = worker_id
        self._port = port
        self._data: dict = {}
        self._trt_data: dict = {}

    def update_data(self, wdata: dict, snapshot: dict) -> None:
        self._data = wdata
        self._trt_data = snapshot
        self.refresh()

    _ZERO = {"avg": 0.0, "p99": 0.0}

    def render(self) -> str:  # type: ignore[override]
        d = self._data
        g = self._trt_data

        inf = d.get("inference_total", 0)
        rps = d.get("req_per_sec", 0.0)
        e2e = d.get("e2e_ms", self._ZERO)
        mcpy = g.get("global_trt_host_copy_ms", self._ZERO)
        h2d = g.get("global_trt_h2d_ms", self._ZERO)
        sem = g.get("global_sem_wait_ms", self._ZERO)
        infer = g.get("global_trt_execute_ms", self._ZERO)
        d2h = g.get("global_trt_d2h_ms", self._ZERO)
        pp = d.get("postprocess_ms", self._ZERO)

        lines = [
            f"[bold cyan]WORKER {self._worker_id}[/bold cyan]  [dim]:{self._port}[/dim]  [yellow]{rps:5.1f}/s [/yellow] [green]{inf:>7,}[/green]",
            "  [dim]         avg     p99[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
            f"  [yellow]MCpy[/yellow]  [white]{_fmt_ms_row(mcpy)}[/white] ms",
            f"  [cyan]H2D[/cyan]   [white]{_fmt_ms_row(h2d)}[/white] ms",
            f"  [red]SWait[/red] [white]{_fmt_ms_row(sem)}[/white] ms",
            f"  [blue]Exec[/blue]  [white]{_fmt_ms_row(infer)}[/white] ms",
            f"  [magenta]D2H[/magenta]   [white]{_fmt_ms_row(d2h)}[/white] ms",
            f"  [green]PostP[/green] [white]{_fmt_ms_row(pp)}[/white] ms",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global stats panel
# ---------------------------------------------------------------------------


class GlobalPanel(Static):
    """Displays global metrics: concurrent inferences, TRT timing."""

    DEFAULT_CSS = """
    GlobalPanel {
        width: 1fr;
        height: 100%;
        border: double $warning;
        padding: 0 1;
        background: $background;
        color: $text;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="global-panel")
        self._data: dict = {}

    def update_data(self, snapshot: dict) -> None:
        self._data = snapshot
        self.refresh()

    _ZERO = {"avg": 0.0, "p99": 0.0}

    def render(self) -> str:  # type: ignore[override]
        d = self._data
        rps = d.get("global_throughput_rps", 0.0)
        e2e = d.get("global_e2e_ms", self._ZERO)
        mcpy = d.get("global_trt_host_copy_ms", self._ZERO)
        h2d = d.get("global_trt_h2d_ms", self._ZERO)
        sem = d.get("global_sem_wait_ms", self._ZERO)
        infer = d.get("global_trt_execute_ms", self._ZERO)
        d2h = d.get("global_trt_d2h_ms", self._ZERO)
        pp = d.get("global_postprocess_ms", self._ZERO)

        lines = [
            f"[bold yellow]GLOBAL METRICS[/bold yellow]  [green]{rps:7.2f} rps[/green]",
            "  [dim]         avg     p99[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
            f"  [yellow]MCpy[/yellow]  [white]{_fmt_ms_row(mcpy)}[/white] ms",
            f"  [cyan]H2D[/cyan]   [white]{_fmt_ms_row(h2d)}[/white] ms",
            f"  [red]SWait[/red] [white]{_fmt_ms_row(sem)}[/white] ms",
            f"  [blue]Exec[/blue]  [white]{_fmt_ms_row(infer)}[/white] ms",
            f"  [magenta]D2H[/magenta]   [white]{_fmt_ms_row(d2h)}[/white] ms",
            f"  [green]PostP[/green] [white]{_fmt_ms_row(pp)}[/white] ms",
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU panel
# ---------------------------------------------------------------------------


class GpuPanel(Widget):
    """Live GPU utilisation sparkline + power bar, styled like WorkerPanel."""

    DEFAULT_CSS = """
    GpuPanel {
        width: 1fr;
        height: 100%;
        border: double $warning;
        padding: 0 1;
        background: $background;
        color: $text;
    }
    GpuPanel > .gpu-header {
        height: 1;
        color: $warning;
        text-style: bold;
    }
    GpuPanel > #gpu-spark {
        height: 1fr;
    }
    GpuPanel > .gpu-summary {
        height: 1;
        color: $text-muted;
    }
    GpuPanel > .gpu-power {
        height: 1;
        color: $text-muted;
    }
    #gpu-spark > .sparkline--max-color {
        color: $warning;
    }
    #gpu-spark > .sparkline--min-color {
        color: $success;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("GPU  —", classes="gpu-header")
        yield Sparkline([], id="gpu-spark", summary_function=max)
        yield Static("", classes="gpu-summary")
        yield Static("", classes="gpu-power")

    def update_data(self, gpu: dict) -> None:
        name = gpu.get("name", "Unknown")
        index = gpu.get("index", 0)
        temp_c = gpu.get("temp_c", 0)
        util_pct = gpu.get("util_pct", 0.0)
        power_w = gpu.get("power_w", 0.0)
        power_limit_w = gpu.get("power_limit_w", 0.0)
        mem_used_mb = gpu.get("mem_used_mb", 0.0)
        mem_total_mb = gpu.get("mem_total_mb", 0.0)
        sparkline_data = gpu.get("sparkline", [])

        self.query_one(".gpu-header", Static).update(
            f"[bold]GPU {index}[/bold]  [dim]{name}[/dim]  [yellow]{temp_c}°C[/yellow]"
        )
        self.query_one("#gpu-spark", Sparkline).data = sparkline_data

        nonzero = [v for v in sparkline_data if v > 0]
        if nonzero:
            summary = (
                f"  now {util_pct:.0f}%  "
                f"avg {sum(nonzero) / len(nonzero):.0f}%  "
                f"peak {max(sparkline_data):.0f}%"
            )
        else:
            summary = ""
        self.query_one(".gpu-summary", Static).update(summary)

        if power_limit_w > 0:
            pct = min(1.0, power_w / power_limit_w)
            bar_w = 10
            filled = round(pct * bar_w)
            bar = "█" * filled + "░" * (bar_w - filled)
            color = "red" if pct >= 0.9 else "yellow"
            mem_gb = f"{mem_used_mb / 1024:.1f}/{mem_total_mb / 1024:.1f}GB"
            pwr_text = (
                f"  Pwr [{color}]{power_w:.0f}W[/{color}]/{power_limit_w:.0f}W "
                f"[{color}]{bar}[/{color}]  Mem {mem_gb}"
            )
        else:
            pwr_text = f"  Pwr {power_w:.0f}W"
        self.query_one(".gpu-power", Static).update(pwr_text)


# ---------------------------------------------------------------------------
# Legend modal
# ---------------------------------------------------------------------------


class LegendPanel(Widget):
    """Color legend for the E2E latency stacked bar chart.

    Occupies the right 1/3 of the latency panel area.  Hidden by default;
    toggled with  l.
    """

    DEFAULT_CSS = """
    LegendPanel {
        width: 1fr;
        height: 1fr;
        border-left: heavy $success;
        padding: 0 1;
        background: $background;
        display: none;
    }
    LegendPanel > .lp-title {
        height: 1;
        color: $success;
        text-style: bold;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(" ▶ STAGE LEGEND", classes="lp-title")
        yield Static("")
        for stage in STAGE_ORDER:
            color = STAGE_COLORS[stage]
            yield Static(f"  [{color}]██[/{color}]  [{color}]{STAGE_ABBREV[stage]}[/{color}]")


# ---------------------------------------------------------------------------
# Tick selector dialog
# ---------------------------------------------------------------------------


class TickSelectorScreen(ModalScreen):
    """Modal dialog for choosing seconds-per-tick."""

    DEFAULT_CSS = """
    TickSelectorScreen {
        align: center middle;
    }
    #tick-dialog {
        width: 44;
        height: auto;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    .dlg-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    .dlg-opt {
        color: $text;
    }
    .dlg-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS = [
        ("1", "pick_1", "1s/tick"),
        ("2", "pick_2", "2s/tick"),
        ("5", "pick_5", "5s/tick"),
        ("0", "pick_10", "10s/tick"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current: int) -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Static(id="tick-dialog"):
            yield Static("SET TICK DURATION", classes="dlg-title")
            yield Static("  [bold]1[/bold]  →  1 second per tick", classes="dlg-opt")
            yield Static("  [bold]2[/bold]  →  2 seconds per tick", classes="dlg-opt")
            yield Static("  [bold]5[/bold]  →  5 seconds per tick", classes="dlg-opt")
            yield Static("  [bold]0[/bold]  →  10 seconds per tick", classes="dlg-opt")
            yield Static(
                f"  [dim]current: {self._current}s/tick — ESC to cancel[/dim]",
                classes="dlg-hint",
            )

    def action_pick_1(self) -> None:
        self.dismiss(1)

    def action_pick_2(self) -> None:
        self.dismiss(2)

    def action_pick_5(self) -> None:
        self.dismiss(5)

    def action_pick_10(self) -> None:
        self.dismiss(10)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Custom footer
# ---------------------------------------------------------------------------


class AppFooter(Static):
    """Footer row: key hints + current tick rate."""

    DEFAULT_CSS = """
    AppFooter {
        height: 1;
        dock: bottom;
        background: #111111;
        color: $text-muted;
        padding: 0 1;
    }
    """

    seconds_per_tick: reactive[int] = reactive(2)
    workers_visible: reactive[bool] = reactive(False)

    def render(self) -> str:
        spt = self.seconds_per_tick
        w_label = "GPU" if self.workers_visible else "Workers"
        return (
            f"[bold white]q[/bold white] Quit  "
            f"[bold white]t[/bold white] Tick  "
            f"[bold white]l[/bold white] Legend  "
            f"[bold white]w[/bold white] {w_label}  "
            f"[dim cyan]{spt}s/tick[/dim cyan]"
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class CondorTUI(App[None]):
    """Condor metrics TUI."""

    TITLE = "CONDOR — Frigate Remote Detector"

    CSS = """
    Screen {
        background: #0d0d0d;
        layers: base;
    }

    #graphs-row {
        height: 1fr;
    }

    #graphs-row GraphPanel {
        width: 1fr;
        height: 1fr;
    }

    #graphs-row #throughput-panel {
        border: heavy $accent;
    }

    #graphs-row #throughput-panel > .title {
        color: $accent;
    }

    #workers-row {
        height: 11;
        layout: horizontal;
    }

    Footer {
        background: #111111;
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("t", "set_tick", "Set Tick"),
        ("l", "legend", "Legend"),
        ("w", "workers", "Workers"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._snapshot: dict = {}
        self._layout_ready = False
        self._num_workers = 0
        self._seconds_per_tick: int = 2
        self._num_ticks: int = 60
        self._stats_writer: asyncio.StreamWriter | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield StatusBanner()
        with Horizontal(id="graphs-row"):
            yield StackedBarPanel()
            yield GraphPanel("THROUGHPUT", "req/s", "throughput-panel")
        with Horizontal(id="workers-row"):
            yield GpuPanel()  # visible by default; workers mounted hidden on first snapshot
        yield AppFooter()

    # ------------------------------------------------------------------
    # Stats reader worker
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _read_stats(self) -> None:
        """Async worker: connects to the stats socket and reads snapshots."""
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
                self._stats_writer = writer
                # Tell the server about our current time config immediately.
                await self._send_time_config()
                async for line in reader:
                    text = line.decode(errors="replace").strip()
                    if not text:
                        continue
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    await self._update_ui(data)
                writer.close()
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                pass
            finally:
                self._stats_writer = None
            self._update_disconnected()
            await asyncio.sleep(2.0)

    def on_mount(self) -> None:
        self._read_stats()

    # ------------------------------------------------------------------
    # Tick config
    # ------------------------------------------------------------------

    async def _send_time_config(self) -> None:
        """Push current window_s / sparkline_len config to the server."""
        w = self._stats_writer
        if w is None or w.is_closing():
            return
        window_s = self._num_ticks * self._seconds_per_tick
        msg = (
            json.dumps({"window_s": window_s, "sparkline_len": self._num_ticks}) + "\n"
        )
        try:
            w.write(msg.encode())
            await w.drain()
        except Exception:
            pass

    def action_set_tick(self) -> None:
        # push_screen_wait requires a worker context — delegate immediately.
        self._open_tick_dialog()

    def action_legend(self) -> None:
        legend = self.query_one(LegendPanel)
        legend.display = not legend.display

    def action_workers(self) -> None:
        gpu_panel = self.query_one(GpuPanel)
        show_workers = gpu_panel.display  # if GPU is visible, switch to workers
        gpu_panel.display = not show_workers
        for wp in self.query(WorkerPanel):
            wp.display = show_workers
        self.query_one(AppFooter).workers_visible = show_workers

    @work
    async def _open_tick_dialog(self) -> None:
        result = await self.push_screen_wait(TickSelectorScreen(self._seconds_per_tick))
        if result is not None:
            self._seconds_per_tick = result
            self.query_one(AppFooter).seconds_per_tick = result
            await self._send_time_config()

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------

    def _update_disconnected(self) -> None:
        try:
            self.query_one(StatusBanner).update_disconnected()
        except Exception:
            pass

    async def _update_ui(self, data: dict) -> None:
        self._snapshot = data
        cfg = data.get("config", {})
        num_workers = cfg.get("num_workers", 1)
        base_port = cfg.get("base_port", 5555)

        # Update header status
        uptime = data.get("uptime_s", 0.0)
        workers_active = data.get("active_workers", 0)
        model_raw = data.get("active_model", "")
        model = Path(model_raw).stem if model_raw else "(none)"

        self.query_one(StatusBanner).update_status(
            uptime, workers_active, num_workers, model
        )

        # Derive num_ticks from the latency panel's content width so the
        # graph X-axis and metric rolling windows stay in sync.
        try:
            lat_panel = self.query_one("#latency-panel", StackedBarPanel)
            # Widget.size in Textual is already the content area (border and
            # padding excluded), so no adjustment needed.
            w = lat_panel.size.width
            if w > 0 and w != self._num_ticks:
                self._num_ticks = w
                await self._send_time_config()
        except Exception:
            pass

        # Update sparklines — trim or left-pad to exactly _num_ticks points so
        # the X-scale is always consistent regardless of uptime or window changes.
        n = self._num_ticks
        lat_data = list(data.get("sparkline_latency", []))
        tput_data = list(data.get("sparkline_throughput", []))
        if len(lat_data) > n:
            lat_data = lat_data[-n:]
        elif len(lat_data) < n:
            lat_data = [0.0] * (n - len(lat_data)) + lat_data
        if len(tput_data) > n:
            tput_data = tput_data[-n:]
        elif len(tput_data) < n:
            tput_data = [0.0] * (n - len(tput_data)) + tput_data

        lat_summary = ""
        if any(v > 0 for v in lat_data):
            nonzero = [v for v in lat_data if v > 0]
            lat_summary = (
                f"  now {lat_data[-1]:.1f}  "
                f"avg {sum(nonzero) / len(nonzero):.1f}  "
                f"peak {max(lat_data):.1f}"
            )

        # Extract and pad per-stage sparkline histories to n ticks
        stages_raw = data.get("sparkline_stages", {})
        stages: dict[str, list[float]] = {}
        for stage in STAGE_ORDER:
            hist = list(stages_raw.get(stage, []))
            if len(hist) > n:
                hist = hist[-n:]
            elif len(hist) < n:
                hist = [0.0] * (n - len(hist)) + hist
            stages[stage] = hist

        self.query_one("#latency-panel", StackedBarPanel).update_data(
            lat_data, stages, n, lat_summary
        )

        tput_summary = ""
        if any(v > 0 for v in tput_data):
            nonzero = [v for v in tput_data if v > 0]
            tput_summary = (
                f"  now {tput_data[-1]:.1f}  "
                f"avg {sum(nonzero) / len(nonzero):.1f}  "
                f"peak {max(tput_data):.1f}"
            )
        self.query_one("#throughput-panel", GraphPanel).update_data(
            tput_data, tput_summary
        )

        # Create worker panels on first snapshot (or if worker count changes)
        workers = data.get("workers", {})
        if not self._layout_ready or self._num_workers != num_workers:
            await self._create_worker_panels(num_workers, base_port)
            self._layout_ready = True
            self._num_workers = num_workers

        # Update per-worker panels — pass full snapshot so workers can read
        # global TRT timing (H2D etc.) which has no per-worker breakdown.
        for wid_str, wdata in workers.items():
            try:
                wid = int(wid_str)
                panel = self.query_one(f"#worker-panel-{wid}", WorkerPanel)
                panel.update_data(wdata, data)
            except Exception:
                pass

        # Update global panel
        try:
            self.query_one("#global-panel", GlobalPanel).update_data(data)
        except Exception:
            pass

        # Update GPU panel
        gpu_data = data.get("gpu")
        if gpu_data:
            spark = list(gpu_data.get("sparkline", []))
            if len(spark) > n:
                spark = spark[-n:]
            elif len(spark) < n:
                spark = [0.0] * (n - len(spark)) + spark
            try:
                self.query_one(GpuPanel).update_data({**gpu_data, "sparkline": spark})
            except Exception:
                pass

    async def _create_worker_panels(self, num_workers: int, base_port: int) -> None:
        """Mount worker panels into the workers-row container."""
        container = self.query_one("#workers-row", Horizontal)
        # Remove old worker/global panels but keep GpuPanel
        for child in list(container.children):
            if not isinstance(child, GpuPanel):
                await child.remove()

        workers_visible = self.query_one(AppFooter).workers_visible
        for i in range(num_workers):
            wp = WorkerPanel(i, base_port + i)
            wp.display = workers_visible
            await container.mount(wp)
        await container.mount(GlobalPanel())  # always visible


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = CondorTUI()
    app.run()


if __name__ == "__main__":
    main()
