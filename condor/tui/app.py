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
from .art import CONDOR_LOGO, build_combined_logo, get_provider_logo, get_bird_frame

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
    "mcpy":  "yellow",
    "h2d":   "cyan",
    "swait": "red",
    "exec":  "blue",
    "d2h":   "magenta",
    "pp":    "green",
}
# Pipeline execution order — determines top-to-bottom stack order in the bar.
STAGE_ORDER: list[str] = ["mcpy", "h2d", "swait", "exec", "d2h", "pp"]
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
    """Format a cur_min_max dict as a fixed-width string.

    Columns: current (no label) | min | max
    All values are zero when the metric has no data.
    """
    return f"{d['cur']:6.1f}  {d['min']:6.1f}  {d['max']:6.1f}"


# ---------------------------------------------------------------------------
# Header widget
# ---------------------------------------------------------------------------


class HeaderWidget(Static):
    """Combined CONDOR logo (left) and provider logo (right) with flying bird animation."""

    DEFAULT_CSS = """
    HeaderWidget {
        height: 8;
        border: heavy $success;
        color: $success;
        background: $background;
    }
    """

    provider: reactive[str] = reactive("", layout=False)

    def __init__(self) -> None:
        super().__init__()
        self._bird_x = 5
        self._bird_y = 1
        self._bird_x_direction = 1
        self._bird_y_direction = 1
        self._animation_tick = 0

    def on_mount(self) -> None:
        """Animation disabled temporarily."""
        pass

    def render(self) -> str:
        condor_lines = CONDOR_LOGO.split("\n")
        provider_lines = get_provider_logo(self.provider).split("\n")

        return build_combined_logo(
            condor_lines,
            provider_lines,
            total_width=self.size.width,
        )

    def watch_provider(self) -> None:
        """Refresh when provider changes."""
        self.refresh()


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar {
        height: 3;
        border: heavy $primary;
        color: $primary;
        content-align: center middle;
        background: $background;
        padding: 0 1;
    }
    """


# ---------------------------------------------------------------------------
# Stacked-bar latency panel
# ---------------------------------------------------------------------------


class StackedBarPanel(Static):
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
    """

    def __init__(self) -> None:
        super().__init__(id="latency-panel")
        self._lat_data: list[float] = []
        self._stages: dict[str, list[float]] = {}
        self._n: int = 60
        self._summary: str = ""

    def update_data(
        self,
        lat_data: list[float],
        stages: dict[str, list[float]],
        n: int,
        summary: str,
    ) -> None:
        self._lat_data = lat_data
        self._stages = stages
        self._n = n
        self._summary = summary
        self.refresh()

    def render(self) -> str:  # type: ignore[override]
        lat = self._lat_data
        stages = self._stages

        # self.size inside render() is INSIDE the border (border already excluded)
        # but still includes padding (0 vertical, 1 left + 1 right).
        bar_h = max(1, self.size.height - 2)   # content_h - title(1) - summary(1)
        bar_w = max(1, self.size.width - 2)     # content_w - padding left(1) - right(1)

        peak = max(lat) if lat else 0.0
        title = f" ▶ E2E LATENCY  [dim]↑ {peak:.0f}[/dim]"

        if not lat:
            return f"{title}\n{self._summary}"

        # Align to rightmost n_cols ticks; left-pad to bar_w with baseline markers
        n_cols = min(bar_w, len(lat))
        offset = len(lat) - n_cols
        lat_slice = lat[offset:]
        n_blank = bar_w - n_cols

        # Grid is always bar_w wide so rendered rows fill the content area exactly
        grid: list[list[str]] = [[""] * bar_w for _ in range(bar_h)]
        # Blank left-padding columns get a baseline marker at the bottom row
        for col in range(n_blank):
            grid[bar_h - 1][col] = _BASELINE_COLOR
        # Real data in rightmost n_cols columns
        for col_idx in range(n_cols):
            col = n_blank + col_idx
            t_idx = offset + col_idx
            vals: dict[str, float] = {
                stage: (stages.get(stage, [])[t_idx]
                        if t_idx < len(stages.get(stage, [])) else 0.0)
                for stage in STAGE_ORDER
            }
            col_colors = _build_column(vals, bar_h, lat_slice[col_idx], peak)
            for row in range(bar_h):
                grid[row][col] = col_colors[row]

        lines = [title]
        for row in grid:
            lines.append(_render_bar_row(row))
        lines.append(f"[dim]{self._summary}[/dim]")
        return "\n".join(lines)


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
    Sparkline > .sparkline--min-color {
        color: $success-darken-3;
    }
    Sparkline > .sparkline--max-color {
        color: $success;
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

    _ZERO = {"cur": 0.0, "min": 0.0, "max": 0.0}

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
            f"[bold cyan]WORKER {self._worker_id}[/bold cyan]  [dim]:{self._port}[/dim]  [yellow]{rps:5.1f} rps[/yellow]",
            f"  Inf   [green]{inf:>7,}[/green]",
            "  [dim]                 min    max[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
            f"  MCpy  [white]{_fmt_ms_row(mcpy)}[/white] ms",
            f"  H2D   [white]{_fmt_ms_row(h2d)}[/white] ms",
            f"  SWait [white]{_fmt_ms_row(sem)}[/white] ms",
            f"  Exec  [white]{_fmt_ms_row(infer)}[/white] ms",
            f"  D2H   [white]{_fmt_ms_row(d2h)}[/white] ms",
            f"  PostP [white]{_fmt_ms_row(pp)}[/white] ms",
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

    _ZERO = {"cur": 0.0, "min": 0.0, "max": 0.0}

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
            "  [dim]                 min    max[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
            f"  MCpy  [white]{_fmt_ms_row(mcpy)}[/white] ms",
            f"  H2D   [white]{_fmt_ms_row(h2d)}[/white] ms",
            f"  SWait [white]{_fmt_ms_row(sem)}[/white] ms",
            f"  Exec  [white]{_fmt_ms_row(infer)}[/white] ms",
            f"  D2H   [white]{_fmt_ms_row(d2h)}[/white] ms",
            f"  PostP [white]{_fmt_ms_row(pp)}[/white] ms",
        ]

        return "\n".join(lines)


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
            yield Static("  [bold]1[/bold]  →  1 second per tick",   classes="dlg-opt")
            yield Static("  [bold]2[/bold]  →  2 seconds per tick",  classes="dlg-opt")
            yield Static("  [bold]5[/bold]  →  5 seconds per tick",  classes="dlg-opt")
            yield Static("  [bold]0[/bold]  →  10 seconds per tick", classes="dlg-opt")
            yield Static(
                f"  [dim]current: {self._current}s/tick — ESC to cancel[/dim]",
                classes="dlg-hint",
            )

    def action_pick_1(self)  -> None: self.dismiss(1)
    def action_pick_2(self)  -> None: self.dismiss(2)
    def action_pick_5(self)  -> None: self.dismiss(5)
    def action_pick_10(self) -> None: self.dismiss(10)
    def action_cancel(self)  -> None: self.dismiss(None)


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

    def render(self) -> str:
        spt = self.seconds_per_tick
        return (
            f"[bold white]q[/bold white] Quit  "
            f"[bold white]t[/bold white] Set Tick  "
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
        height: 1fr;
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
        yield HeaderWidget()
        yield StatusBar("● CONNECTING…", id="status-bar")
        with Horizontal(id="graphs-row"):
            yield StackedBarPanel()
            yield GraphPanel("THROUGHPUT", "req/s", "throughput-panel")
        with Horizontal(id="workers-row"):
            pass  # Worker panels mounted dynamically on first snapshot
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
        msg = json.dumps({"window_s": window_s, "sparkline_len": self._num_ticks}) + "\n"
        try:
            w.write(msg.encode())
            await w.drain()
        except Exception:
            pass

    def action_set_tick(self) -> None:
        # push_screen_wait requires a worker context — delegate immediately.
        self._open_tick_dialog()

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
            self.query_one("#status-bar", StatusBar).update(
                "[bold red]● DISCONNECTED[/bold red]  "
                f"[dim]waiting for condor server at {SOCKET_PATH}…[/dim]"
            )
        except Exception:
            pass

    async def _update_ui(self, data: dict) -> None:
        self._snapshot = data
        cfg = data.get("config", {})
        provider = cfg.get("provider", "")
        num_workers = cfg.get("num_workers", 1)
        base_port = cfg.get("base_port", 5555)

        # Update header provider logo
        header = self.query_one(HeaderWidget)
        if header.provider != provider:
            header.provider = provider

        # Update status bar
        uptime = data.get("uptime_s", 0.0)
        workers_active = data.get("active_workers", 0)
        model_raw = data.get("active_model", "")
        model = Path(model_raw).stem if model_raw else "(none)"
        concurrent = data.get("inference_concurrent", 0)
        rps = data.get("global_throughput_rps", 0.0)

        status_text = (
            f"[bold green]● ONLINE[/bold green]  "
            f"⏱ [cyan]{_fmt_time(uptime)}[/cyan]  "
            f"⚙ [yellow]{workers_active}/{num_workers} workers[/yellow]  "
            f"📦 [white]{model}[/white]  "
            f"[magenta]{provider or '—'}[/magenta]"
        )
        self.query_one("#status-bar", StatusBar).update(status_text)

        # Derive num_ticks from the latency panel's content width so the
        # graph X-axis and metric rolling windows stay in sync.
        try:
            lat_panel = self.query_one("#latency-panel", StackedBarPanel)
            # Content width = outer_w - border(2) - padding(2)
            w = lat_panel.size.width - 4
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
                f"avg {sum(nonzero)/len(nonzero):.1f}  "
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
                f"avg {sum(nonzero)/len(nonzero):.1f}  "
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

    async def _create_worker_panels(self, num_workers: int, base_port: int) -> None:
        """Mount worker panels into the workers-row container."""
        container = self.query_one("#workers-row", Horizontal)
        for child in list(container.children):
            await child.remove()

        for i in range(num_workers):
            await container.mount(WorkerPanel(i, base_port + i))
        await container.mount(GlobalPanel())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = CondorTUI()
    app.run()


if __name__ == "__main__":
    main()
